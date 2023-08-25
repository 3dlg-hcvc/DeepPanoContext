import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from glob import glob
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
import matplotlib.pyplot as plt
import cv2

from gibson2.utils.assets_utils import get_ig_scene_path, get_cubicasa_scene_path, get_3dfront_scene_path

from .mesh_utils import load_mesh
from external.HorizonNet.dataset import cor_2_1d, find_occlusion
from .transform_utils import bdb3d_from_front_face, IGTransform, interpolate_line
from utils.image_utils import show_image


def plot_layout(layout, cameras=None):
    def plot_poly(poly):
        plt.plot(*poly.exterior.xy)
        for i in poly.interiors:
            plt.plot(*i.xy)

    if isinstance(layout, Polygon):
        plot_poly(layout)
    else:
        for poly in layout:
            plot_poly(poly)

    def plot_camera(camera):
        pos = camera['pos'][:2]
        target = camera['target'][:2]
        plt.arrow(*pos, target[0] - pos[0], target[1] - pos[1], width=0.1)
        plt.text(*pos, camera['name'], fontsize=12)

    if cameras:
        if isinstance(cameras, dict):
            plot_camera(cameras)
        else:
            for camera in cameras:
                plot_camera(camera)

    plt.axis('equal')
    plt.show()


def scene_layout_from_mesh(args):
    scene_name, scene_source = args.scene_name, args.scene_source
    if scene_source == "IG":
        scene_dir = get_ig_scene_path(scene_name)
    elif scene_source == "CUBICASA":
        scene_dir = get_cubicasa_scene_path(scene_name)
    else:
        scene_dir = get_3dfront_scene_path(scene_name)

    # read wall bbox
    walls = os.path.join(scene_dir, 'shape', 'collision', 'wall_*.obj')
    walls = glob(walls)
    try:
        walls = [load_mesh(wall) for wall in walls]
    except Exception as err:
        print(f"Error: '{err}' while loading walls for {scene_name}")
        return
    # walls_union = trimesh.boolean.union(walls)
    # trimesh.exchange.export.export_mesh(walls_union, os.path.join(input_folder, "union.obj"))
    assert all(wall.is_watertight for wall in walls), f"Not all walls of {scene_name} are watertight"

    # get 2D poly of walls
    wall_height = max([wall.extents[-1] for wall in walls])
    wall2d = []
    for wall in walls:
        wall = wall.bounding_box_oriented
        wall_edges = np.array(wall.vertices[wall.edges_unique])
        is_top = np.all(np.abs(wall_edges[:, :, -1] - wall_height) < 0.001, -1)
        wall_edges = wall_edges[is_top, :, :2]
        length_edges = np.linalg.norm(wall_edges[:, 0, :] - wall_edges[:, 1, :], ord=2, axis=-1).squeeze()
        lengths = np.unique(length_edges)
        lengths.sort()
        lengths = lengths[1:][(lengths[1:] - lengths[:-1]) > 1e-5]
        if len(lengths) != 2:
            continue
        wall_edges = wall_edges[np.abs(length_edges - lengths[0]) < 1e-5, :, :]
        assert len(wall_edges) == 2
        i = np.argmin(np.linalg.norm(wall_edges[1] - wall_edges[0, 1], ord=2, axis=-1))
        poly = Polygon([wall_edges[0, 0], wall_edges[0, 1], wall_edges[1, i], wall_edges[1, 1 - i]])
        wall2d.append(poly)

    # generate scene layout from poly
    wall2d = [wall.buffer(0.05) for wall in wall2d] # dilate to close wall polygon corners
    wall2d = shapely.ops.cascaded_union(wall2d) # get union of walls to get room layout
    # plot_layout(wall2d)
    if isinstance(wall2d, MultiPolygon): # remove small disconnected wall structure like pillars
        areas = [wall.area for wall in wall2d]
        wall2d = wall2d[(areas.index(max(areas)))]
    wall2d = wall2d.buffer(-0.05) # erose to recover wall thickness
    rooms = [Polygon(r).simplify(0.1) for r in wall2d.interiors] # merge to avoid replicated corners in final layout

    return {'rooms': rooms, 'height': wall_height}


def scene_layout_from_rlsd_arch(args):
    scene_name, _ = args.scene_name, args.scene_source
    house_id = scene_name.split('_')[0]
    
    arch = json.load(open(f"/project/3dlg-hcvc/rlsd/data/mp3d/arch_refined_clean/{house_id}.arch.json"))
    elements = arch["elements"]
    regions = arch["regions"]
    
    rooms = {}
    types = ['Floor', 'Ceiling', 'Wall']
    region_walls = []
    for ele in elements:
        room_id = ele['roomId']
        level = int(ele['roomId'].split('_')[0])
        ele_type = ele['type']
        if ele_type not in types: continue
        if room_id not in rooms:
            rooms[room_id] = {t: [] for t in types}
        rooms[room_id][ele_type].append(ele)
        if ele_type == 'Wall':
            region_walls.extend(ele['points'])
    rooms[room_id]["region_walls"] = np.unique(np.asarray(region_walls), axis=0)
    for region in regions:
        region_xy = np.asarray(region["points"])[:, :-1]
        region_dict = {
            "id": region["id"],
            "level": region["level"],
            "type": region["type"],
            "wall_height": region["height"],
            "floor_height": region["points"][0][-1],
            "room": Polygon(region_xy),
        }
        assert region["id"] in rooms
        rooms[region["id"]].update(region_dict)
    rooms_bounds = shapely.ops.cascaded_union([r['room'] for r in rooms.values()]).bounds
    rooms_height = max([r['wall_height']+r['floor_height'] for r in rooms.values()]) - min([r['floor_height'] for r in rooms.values()])
    rooms_scale = (rooms_bounds[2]-rooms_bounds[0], rooms_bounds[3]-rooms_bounds[1], rooms_height)

    return rooms, rooms_scale


def scene_layout_from_scene(scene):
    # show_image(scene.room_sem_map)
    room_ids = np.unique(scene.room_sem_map).tolist()
    room_ids.remove(0)
    rooms = []
    for i_room in room_ids:
        room_mask = scene.room_sem_map == i_room
        contours, hierarchy = cv2.findContours(
            room_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i_max = np.argmax([cv2.contourArea(contour) for contour in contours])
        contour = contours[i_max][:, 0, :]
        room = scene.seg_map_to_world(contour[:, ::-1])
        rooms.append(Polygon(room))
    # plot_layout(rooms)
    return {'rooms': rooms}


def room_layout_from_scene_layout(camera, scene_layout):
    # find which room in
    camera_point = Point(*camera['pos'][:2])
    in_wall = True
    for room_layout in scene_layout['rooms']:
        if room_layout.contains(camera_point):
            # plot_layout(r, camera)
            # sort boundary points in clockwise order
            room_layout = shapely.geometry.polygon.orient(room_layout, -1)
            boundary = np.array(room_layout.boundary.xy)[:, :-1].T
            assert np.all(np.linalg.norm((np.roll(boundary, 1, 0) - boundary), ord=2, axis=1) > 0.01), \
                'neighboring corners are too close'
            in_wall = False
            break
    if in_wall:
        print("in wall")
        return

    nearest_point, _ = shapely.ops.nearest_points(room_layout.boundary, camera_point)
    distance_wall = camera_point.distance(nearest_point)
    if distance_wall < 0.5:
        print(f"too close ({distance_wall:.3f} < 0.5) to wall")
        return
    if not room_layout.is_valid:
        print("not valid")
        return

    return {'room': room_layout, 'height': scene_layout['height']}


def room_layout_from_rlsd_scene(camera, rooms, panos, plot_path):
    cam_id = camera["id"]
    cam_point = Point(*camera["pos"][:2])
    cam_height = camera["pos"][-1]
    room_id = f"{panos[cam_id]['level_id']}_{panos[cam_id]['region_index']}"
    if not rooms[room_id]["room"].contains(cam_point):
        outside = True
        for room in rooms.values():
            ceil_height = room['wall_height'] + room['floor_height']
            if cam_height > room['floor_height'] and cam_height <= ceil_height and room["room"].contains(cam_point):
                room_id = room["id"]
                outside = False
                break
        if outside:
            return None, None # camera outside rooms -> invalid -> return None
    room = rooms[room_id]
    layout2d = room["room"]
    level = room["level"]
    plot_2d_regions_layout(rooms, room_id, level, camera, os.path.join(plot_path, "regions.png"))
    plot_2d_walls_layout(rooms, level, camera, os.path.join(plot_path, "walls.png"))

    # sort boundary points in clockwise order
    layout2d = shapely.geometry.polygon.orient(layout2d, -1)
    room["room"] = layout2d
    boundary = np.array(layout2d.boundary.xy)[:, :-1].T
    if np.all(np.linalg.norm((np.roll(boundary, 1, 0) - boundary), ord=2, axis=1) <= 0.01):
        print(f'{cam_id} neighboring corners are too close')

    nearest_point, _ = shapely.ops.nearest_points(layout2d.boundary, cam_point)
    distance_wall = cam_point.distance(nearest_point)

    return room, distance_wall


def plot_2d_regions_layout(rooms, room_id, level=0, cameras=[], output_path=None):
    if isinstance(cameras, dict):
        cameras = [cameras]
    _, ax = plt.subplots(figsize=(8,8))
    for id, room in rooms.items():
        if room["level"] == level:
            ax.plot(*room["room"].exterior.xy)
            if id == room_id:
                ax.fill(*room["room"].exterior.xy, "lightgreen")  
    for cam in cameras:
        pos = cam["pos"][:2]
        view_dir = cam["view_dir"][:2]
        cam_point = Point(*pos).buffer(0.2)
        ax.plot(*cam_point.exterior.xy) 
        ax.arrow(*pos, *view_dir, width=0.1)
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()


def plot_2d_walls_layout(rooms, level=0, cameras=[], output_path=None):
    if isinstance(cameras, dict):
        cameras = [cameras]
    level_rooms = {id: r for id, r in rooms.items() if int(id.split('_')[0]) == level}
    _, ax = plt.subplots(figsize=(8,8))
    colors = plt.cm.jet(np.linspace(0, 1, len(level_rooms)))
    for idx, (room_id, room) in enumerate(level_rooms.items()):
        for wall in room['Wall']:
            wall_points = np.asarray(wall["points"])[:, :-1]
            ax.plot(wall_points[:,0], wall_points[:,1], color=colors[idx])
    for cam in cameras:
        pos = cam["pos"][:2]
        view_dir = cam["view_dir"][:2]
        cam_point = Point(*pos).buffer(0.2)
        ax.plot(*cam_point.exterior.xy) 
        ax.arrow(*pos, *view_dir, width=0.1)
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()


def manhattan_pix_layout_from_room_layout(camera, room_layout):
    if room_layout is None:
        return

    # generate pano Manhattan layout from room layout
    height, width = camera['height'], camera['width']
    boundary = np.array(room_layout['room'].boundary.xy)[:, :-1].T
    camera_height = camera['pos'][-1]
    directions = []
    camera_point = np.array(camera['pos'][:2])
    front = np.arctan2(*(np.array(camera['target'][:2]) - camera_point))
    points = []
    for p in boundary:
        direction = np.mod(np.arctan2(*(p - camera_point)) - front + np.pi, np.pi * 2)
        directions.append(direction)
        x = direction / (2 * np.pi) * width
        dis = np.linalg.norm(p - camera_point, 2)
        pitch = np.arctan2(room_layout['height'] - camera_height, dis)
        y_top = (np.pi / 2 - pitch) / np.pi * height
        points.append([x, y_top])
        pitch = np.arctan2(camera_height, dis)
        y_down = (pitch + np.pi / 2) / np.pi * height
        points.append([x, y_down])
    points = np.array(points)
    i_first = directions.index(min(directions))
    points = np.roll(points, -i_first * 2, axis=0)
    points_unique = np.unique(points, axis=0)
    if len(points_unique) < len(points):
        print("duplicate points")
        return
    xs = points[::2, 0].astype(np.int)
    xs_unique = np.unique(xs)
    if len(xs_unique) < len(xs):
        print("duplicate x")
        return

    return points.astype(np.int32)


def manhattan_pix_layout_from_rlsd_room(camera, room, room_mode, full_task_id, issues):
    if room is None:
        return
    if room_mode == "regions":
        boundary = np.array(room['room'].boundary.xy)[:, :-1].T
    elif room_mode == "walls":
        boundary = room['region_walls'][:, :-1]
    else:
        raise NotImplemented
        
    # generate pano Manhattan layout from room layout
    height, width = camera['height'], camera['width']
    camera_height = camera['pos'][-1] - room["floor_height"]
    directions = []
    camera_point = np.array(camera['pos'][:2])
    front = np.arctan2(*np.array(camera['view_dir'][:2]))
    points = []
    for p in boundary:
        direction = np.mod(np.arctan2(*(p - camera_point)) - front + np.pi, np.pi * 2)
        directions.append(direction)
        x = direction / (2 * np.pi) * width
        dis = np.linalg.norm(p - camera_point, 2)
        pitch = np.arctan2(room['wall_height'] - camera_height, dis)
        y_top = (np.pi / 2 - pitch) / np.pi * height
        points.append([x, y_top])
        pitch = np.arctan2(camera_height, dis)
        y_down = (pitch + np.pi / 2) / np.pi * height
        points.append([x, y_down])
    points = np.array(points)
    i_first = directions.index(min(directions))
    points = np.roll(points, -i_first * 2, axis=0)
    points_unique = np.unique(points, axis=0)
    if len(points_unique) < len(points):
        issues["duplicate_points"].append(full_task_id)
        print(f"{full_task_id} duplicate points")
        # return
    xs = points[::2, 0].astype(int)
    xs_unique = np.unique(xs)
    if len(xs_unique) < len(xs):
        issues["duplicate_x"].append(full_task_id)
        print(f"{full_task_id} duplicate x")
        # return

    return points.astype(np.int32)


def cuboid_world_layout_from_room_layout(room_layout):
    # generate cuboid layout in the world frame from bounding box
    # plot_layout(r, camera)
    bdb2d = shapely.geometry.box(*room_layout['room'].bounds, ccw=True)
    # plot_layout(bdb2d, camera)
    room_layout = room_layout.copy()
    room_layout['room'] = bdb2d
    bbox3d = manhattan_world_layout_from_room_layout(room_layout)
    return bbox3d


def manhattan_world_layout_from_room_layout(room_layout):
    # generate manhattan layout in the world frame from bondary
    boundary = np.array(room_layout['room'].boundary.xy, dtype=np.float32)[:, :-1].T
    boundary_l = np.pad(boundary, [[0, 0], [0, 1]], 'constant', constant_values=0)
    boundary_h = np.pad(boundary, [[0, 0], [0, 1]], 'constant', constant_values=room_layout['height'] if 'height' in room_layout else room_layout['wall_height'])
    world_layout = np.concatenate([boundary_l, boundary_h], 0)
    return world_layout


def horizon_layout_gt_from_scene_data(data):
    cor = data['layout']['manhattan_pix']
    camera = data['camera']

    # Detect occlusion
    occlusion = find_occlusion(cor[::2].copy(), w=camera['width'], h=camera['height']).repeat(2)

    # Prepare 1d ceiling-wall/floor-wall boundary
    bon = cor_2_1d(cor, camera['height'], camera['width'])

    # Prepare 1d wall-wall probability
    corx = cor[~occlusion, 0]
    dist_o = cdist(corx.reshape(-1, 1), np.arange(camera['width']).reshape(-1, 1), p=1)
    dist_r = cdist(corx.reshape(-1, 1), np.arange(camera['width']).reshape(-1, 1) + camera['width'], p=1)
    dist_l = cdist(corx.reshape(-1, 1), np.arange(camera['width']).reshape(-1, 1) - camera['width'], p=1)
    dist = np.min([dist_o, dist_r, dist_l], 0)
    nearest_dist = dist.min(0)
    y_cor = (0.96 ** nearest_dist).reshape(1, -1)

    return {'bon': bon.astype(np.float32), 'cor': y_cor.astype(np.float32)}


def wall_bdb3d_from_manhattan_world_layout(layout, thickness=0.2):
    n_walls = len(layout) // 2
    walls = []
    for i_wall in range(n_walls):
        front_face = layout[(i_wall,
                             np.mod(i_wall + 1, n_walls),
                             i_wall + n_walls,
                             np.mod(i_wall + 1, n_walls) + n_walls), :]
        bdb3d_wall = bdb3d_from_front_face(front_face, thickness)
        walls.append({'bdb3d': bdb3d_wall})
    return walls


def wall_contour_from_manhattan_pix_layout(layout, transform: IGTransform):
    n_walls = len(layout) // 2
    walls = []
    for i_wall in range(n_walls):
        front_face = layout[(i_wall * 2 + 1,
                             i_wall * 2,
                             np.mod(i_wall * 2 + 2, len(layout)),
                             np.mod(i_wall * 2 + 3, len(layout))), :]

        # generate contour from front face
        contour = []
        front_face_3d = transform.campix23d(front_face, 1.)
        for p1, p2 in zip(front_face_3d, np.roll(front_face_3d, -1, axis=0)):
            contour.append(interpolate_line(p1, p2)[:-1])
        contour = np.concatenate(contour, 0)
        contour = transform.cam3d2pix(contour)

        # extend contour points divided by edge out of right edge
        outside = False
        contour_ext = []
        for p1, p2 in zip(contour, np.roll(contour, -1, axis=0)):
            if np.abs(p2[0] - p1[0]) > transform.camera['width'] / 2:
                outside = not outside
            if outside:
                p2[0] += transform.camera['width']
            contour_ext.append(p2)
        contour = np.stack(contour_ext)
        poly = Polygon(contour)
        # plt.plot(*poly.exterior.xy)
        # plt.axis('equal')
        # plt.show()

        walls.append({'contour': {
            'x': contour[..., 0].astype(np.int32),
            'y': contour[..., 1].astype(np.int32),
            'area': float(poly.area)
        }})
    return walls


def manhattan_world_layout_from_pix_layout(scene, camera_height=1.6):
    manhattan_pix = scene['layout']['manhattan_pix']
    corners_rad = scene.transform.campix2rad(manhattan_pix)
    corners_lon, corners_lat = corners_rad.T
    ceil_corners_lat, floor_corners_lat = corners_lat[::2], corners_lat[1::2]
    floor_corners_dis = camera_height / np.sin(floor_corners_lat)
    wall_corners_dis = camera_height / np.tan(floor_corners_lat)
    ceil_corners_dis = wall_corners_dis / np.cos(ceil_corners_lat)
    corners_rad = np.flip(corners_rad.reshape(-1, 2, 2).transpose(1, 0, 2), 0).reshape(-1, 2)
    corners_dis = np.concatenate([floor_corners_dis, ceil_corners_dis])
    world_layout = scene.transform.camrad2world(corners_rad, corners_dis)
    return world_layout


def manhattan_2d_from_manhattan_world_layout(layout):
    n_walls = len(layout) // 2
    layout_2d = layout[:n_walls, :2]
    return layout_2d


def manhattan_world_layout_info(layout):
    layout_z = layout[:, -1]
    layout_2d = manhattan_2d_from_manhattan_world_layout(layout)
    layout_poly = Polygon(layout_2d)
    floor = layout_z.min()
    ceil = layout_z.max()
    return {'ceil': ceil, 'floor': floor, 'layout_poly': layout_poly, 'layout_2d': layout_2d}


def layout_line_segment_indexes(N):
    layout_lines = [[i, (i + 1) % N] for i in range(N)] + \
                   [[i + N, (i + 1) % N + N] for i in range(N)] + \
                   [[i, i + N] for i in range(N)]
    return layout_lines
