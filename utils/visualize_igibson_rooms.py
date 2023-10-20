import os
import json
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from copy import deepcopy

from utils.igibson_utils import IGScene
from models.pano3d.dataloader import SceneDataset
from utils.visualize_utils import IGVisualizer
from utils.relation_utils import RelationOptimization, relation_from_bins
from utils.basic_utils import read_pkl
from utils.render_layout_bdb3d import render_view
from utils.mesh_utils import MeshIO, write_ply_rgb_face, create_layout_mesh, create_bdb3d_mesh
from models.eval_metrics import AverageMeter, AveragePrecisionMeter, BinaryClassificationMeter, ClassMeanMeter
from configs.data_config import igibson_colorbox


data_dir = "/project/3dlg-hcvc/rlsd/data/psu/igibson"
out_dir = "/project/3dlg-hcvc/rlsd/data/psu/igibson_rooms"
scene_names = [f for f in os.listdir(data_dir) if not f.endswith(".json")]


def plot_ig_rooms_layout(rooms, output_path=None):
    _, ax = plt.subplots(figsize=(8,8))
    for room_name, room in rooms.items():
        ax.plot(*room.exterior.xy)
        ax.text(*list(room.centroid.coords)[0], s=room_name)
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()


def merge_layout_bdb3d_mesh(data, objs, colorbox=None, separate=False, camera_color=None, layout_color=None, texture=False, filename=None):
    mesh_io = MeshIO()
    
    for k, obj in enumerate(objs):
        bdb3d = obj['bdb3d']
        mesh_world = create_bdb3d_mesh(bdb3d, radius=0.015)
        mesh_io[k] = mesh_world

    # add layout mesh
    if layout_color is not None:
        layout_mesh = create_layout_mesh(data, color=layout_color, texture=texture)
        if layout_mesh is not None:
            mesh_io['layout_mesh'] = layout_mesh
    
    all_verts, all_faces, all_colors = [], [], []
    for k, m in mesh_io.items():
        if k == 'camera': color = camera_color
        elif k == 'layout_mesh': color = layout_color
        else: 
            color = colorbox[objs[k]['label']]
        cur_num_verts = len(all_verts)
        all_verts.extend(m.vertices.tolist())
        all_faces.extend((m.faces + cur_num_verts).tolist())
        all_colors.extend([color] * len(m.vertices))
    
    if filename is not None:
        write_ply_rgb_face(np.array(all_verts), np.array(all_colors), np.array(all_faces), filename)

    if separate:
        return mesh_io

    return mesh_io.merge()


def evaluate_collision(data, room_name, metric, metric_rooms):
    # generate relation between estimated objects and layout
    # 0.1m of toleration distance when measuring collision
    relation_optimization = RelationOptimization(expand_dis=0.1)
    data['objs'] = deepcopy(data['objs'])
    rel_scene = IGScene(data)
    relation_optimization.generate_relation(rel_scene)
    relation = rel_scene['relation']

    # collision metrics
    metric_rooms[room_name]['collision_pairs'].append(relation['obj_obj_col'].sum() / 2)
    metric_rooms[room_name]['collision_objs'].append(relation['obj_obj_col'].any(axis=0).sum())
    metric_rooms[room_name]['collision_walls'].append(relation['obj_wall_col'].any(axis=-1).sum())
    metric_rooms[room_name]['collision_ceil'].append(sum(o['ceil_col'] for o in rel_scene['objs']))
    metric_rooms[room_name]['collision_floor'].append(sum(o['floor_col'] for o in rel_scene['objs']))
    
    metric['collision_pairs'].append(relation['obj_obj_col'].sum() / 2)
    metric['collision_objs'].append(relation['obj_obj_col'].any(axis=0).sum())
    metric['collision_walls'].append(relation['obj_wall_col'].any(axis=-1).sum())
    metric['collision_ceil'].append(sum(o['ceil_col'] for o in rel_scene['objs']))
    metric['collision_floor'].append(sum(o['floor_col'] for o in rel_scene['objs']))

metric = defaultdict(AverageMeter)

for scene_name in tqdm(scene_names):
    scene_dir = os.path.join(out_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    scene_polygons = dict()
    scene_rooms = defaultdict(list)
    scene_layouts = defaultdict(dict)
    scene_objs = defaultdict(dict)
    metric_rooms = defaultdict(lambda: ClassMeanMeter(AverageMeter))
    for i in range(100):
        camera_id = f"{i:05d}"
        pkl_file = os.path.join(data_dir, scene_name, camera_id, 'data.pkl')
        data = read_pkl(pkl_file)
        room_name = data['room']
        scene_rooms[room_name].append(camera_id)
        if room_name not in scene_polygons:
            scene_layouts[room_name]["layout"] = data["layout"]
            corners = data["layout"]["manhattan_world"]
            room = Polygon(corners[:(len(corners)//2), :2])
            scene_polygons[room_name] = room
        scene_objs[room_name].update({obj['id']: obj for obj in data['objs'] if obj['id'] not in scene_objs[room_name]})
        evaluate_collision(data.copy(), room_name, metric, metric_rooms)
    
    plot_ig_rooms_layout(scene_polygons,
                         os.path.join(scene_dir, "rooms.png"))
    with open(os.path.join(scene_dir, "cameras.json"), "w") as f:
        json.dump(scene_rooms, f, indent=4)
        
    for room_name in metric_rooms:
        for k in metric_rooms[room_name]:
            metric_rooms[room_name][k] = metric_rooms[room_name][k]()
    with open(os.path.join(scene_dir, "collisions.json"), "w") as f:
        json.dump(metric_rooms, f, indent=4)
        
    for room_name in scene_layouts:
        room_dir = os.path.join(scene_dir, room_name)
        os.makedirs(room_dir, exist_ok=True)
        _ = merge_layout_bdb3d_mesh(
            scene_layouts[room_name],
            list(scene_objs[room_name].values()),
            colorbox=igibson_colorbox * 255,
            separate=False,
            # camera_color=(29, 203, 224),
            layout_color=(255, 69, 80),
            texture=False,
            filename=os.path.join(room_dir, 'layout_bdb3d.ply')
        )
        render_view(os.path.join(room_dir, 'layout_bdb3d.ply'),
                    os.path.join(room_dir, 'layout_bdb3d.png'))

for k in metric:
    metric[k] = metric[k]()
with open(os.path.join(out_dir, "collisions.json"), "w") as f:
    json.dump(metric, f, indent=4)