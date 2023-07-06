import os
import json
import argparse
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import torch

os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import trimesh
import pyrender
from pyrender import RenderFlags
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer


def points2bdb2d(points):
    points = np.stack([points['x'], points['y']]).T if isinstance(points, dict) else points
    if isinstance(points, torch.Tensor):
        xy_max = torch.max(points, -2)[0]
        xy_min = torch.min(points, -2)[0]
    else:
        xy_max = points.max(-2)
        xy_min = points.min(-2)
    return {
        'x1': xy_min[..., 0],
        'x2': xy_max[..., 0],
        'y1': xy_min[..., 1],
        'y2': xy_max[..., 1]
    }


def seg2obj(seg, i_obj, camera=None):
    """
    Extract contour and bounding box/fov from instance segmentation image.

    Parameters
    ----------
    seg: H x W numpy array of instance segmentation image
    i_obj: instance ID

    Returns
    -------
    dict of object contour and 2D bounding box: dict{
        'bfov': {'lon': float, 'lat': float, 'x_fov': float, 'y_fov': float} in rad
        'bdb2d': {'x1': int, 'x2': int, 'y1': int, 'y2': int} in pixel
        'contour': {'x': 1-d numpy array, 'y': 1-d numpy array, 'area': float} in pixel
    }

    definition of output pixel coordinate:
    x: (left) 0 --> width - 1 (right)
    y: (up) 0 --> height - 1 (down)

    definition of longitude and latitude in radiation:
    longitude: (left) -pi -- 0 --> +pi (right)
    latitude: (up) -pi/2 -- 0 --> +pi/2 (down)
    """

    height, width = seg.shape[:2]
    pano = camera is None or 'K' not in camera
    if pano:
        # if is panorama, repeat image along x axis to connect segmentation mask divided by edge
        seg = np.tile(seg, 2)

    # find and sort contours
    if isinstance(i_obj, list):
        obj_mask = np.isin(seg, i_obj)
    else:
        obj_mask = seg == i_obj
    contours, hierarchy = cv2.findContours(
        obj_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = [cv2.contourArea(contour) for contour in contours]
    contours = [x for _, x in sorted(zip(area, contours), key=lambda pair: pair[0], reverse=True)]
    area = sorted(area, reverse=True)

    if pano:
        # if is panorama, consider objects on edge
        if len(area) > 1 and abs(area[0] - area[1]) < 1:
            # if object is not on the edge, choose the left contour
            contour_a, contour_b = contours[0][:, 0, :], contours[1][:, 0, :]
            contour = contour_a if np.min(contour_a[:, 0]) < np.min(contour_b[:, 0]) else contour_b
        elif len(area) == 0:
            return
        else:
            # if object is on the edge, choose the largest contour
            contour = contours[0][:, 0, :]
    else:
        # if is perspective camera, choose the largest contour
        contour = contours[0][:, 0, :]

    # from contour to bdb2d/bfov
    bdb2d = points2bdb2d(contour)
    bdb2d = {k: int(v) for k, v in bdb2d.items()}
    contour = {
        'x': contour[..., 0].astype(np.int32),
        'y': contour[..., 1].astype(np.int32),
        'area': float(area[0])
    }
    # bfov = contour2bfov(contour, height, width, camera)

    return {
        # 'bfov': bfov,
        'bdb2d': bdb2d,
        'contour': contour
    }


def render_view(args):
    output_folder = os.path.join(args.output, *args.object.split('/')[-2:])
    obj_category = output_folder.split('/')[-2]
    # output_folder = './demo_render'
    os.makedirs(output_folder, exist_ok=True)

    # args.object_path = f"/datasets/internal/models3d/wayfair/wayfair_models_cleaned/{shape_id}/{shape_id}.glb"
    obj = trimesh.load(args.object_path)
    center = np.mean(obj.bounds, axis=0)
    scale = 1.0 / float(max(obj.bounds[1] - obj.bounds[0]))
    center_mat = np.array([
    [1, 0, 0, -center[0]],
    [0.0, 1, 0.0, -center[1]],
    [0.0, 0.0, 1, -center[2]],
    [0.0,  0.0, 0.0, 1.0]
    ])
    norm_mat = np.array([
        [scale, 0, 0, 0],
        [0.0, scale, 0.0, 0],
        [0.0, 0.0, scale, 0],
        [0.0,  0.0, 0.0, 1.0]
    ])
    obj.apply_transform(np.matmul(norm_mat, center_mat))
    if not isinstance(obj, trimesh.Scene):
        obj = trimesh.Scene(obj)
    scene = pyrender.Scene.from_trimesh_scene(obj, ambient_light=[0.1, 0.1, 0.1])
    
    # img = cv2.imread("/local-scratch/qiruiw/research/DeepPanoContext/data/gibson2/ig_dataset/scenes/background/palermo_sidewalk.jpg")
    bg_img = np.asarray(Image.open("/local-scratch/qiruiw/research/DeepPanoContext/ballroom_2k.png"))
    sphere_trimesh = trimesh.load_mesh("/local-scratch/qiruiw/research/rlsd/evaluation/conf/sphere/sphere.obj")
    sphere_scale = np.array([
        [10, 0, 0, 0],
        [0.0, 10, 0.0, 0],
        [0.0, 0.0, 10, 0],
        [0.0,  0.0, 0.0, 1.0]
    ])
    sphere_trimesh.apply_transform(sphere_scale)
    texture = pyrender.Texture(source=bg_img, source_channels="RGB", 
                               sampler=pyrender.Sampler(magFilter=pyrender.constants.GLTF.NEAREST, 
                                                        minFilter=pyrender.constants.GLTF.NEAREST))
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.,1.,1.,1.],baseColorTexture=texture)
    bg_mesh = pyrender.Mesh.from_trimesh(sphere_trimesh, material=material)
    # scene = pyrender.Scene(ambient_light=[1., 1., 1.])
    # scene.add(bg_mesh)

    # cam = PerspectiveCamera(yfov=(np.pi / 2.0))
    direc_l = DirectionalLight(color=np.ones(3), intensity=0.5)
    spot_l = SpotLight(color=np.ones(3), intensity=1.0,
                    innerConeAngle=np.pi/8, outerConeAngle=np.pi/3)
    # point_l = PointLight(color=np.ones(3), intensity=10.0)
    r = OffscreenRenderer(viewport_width=512, viewport_height=512)

    for i_render in range(args.renders):
        # randomize light direction
        # renderer.set_light_position_direction(((np.random.random(3) - 0.5) * 10 + 5).tolist(), [0, 0, 0])
        l_dis = np.random.random() * 0.5 + 1
        azim = np.random.random() * np.pi * 2
        elev = np.random.random() * np.pi
        l_pose = np.eye(4)
        y = l_dis * np.sin(elev)
        x = l_dis * np.cos(elev) * np.sin(azim)
        z = l_dis * np.cos(elev) * np.cos(azim)
        l_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        l_pose[:3, :3] = np.matmul(roty, rotx)
        spot_l_node = scene.add(spot_l, pose=l_pose)
        direc_l_node = scene.add(direc_l, pose=l_pose)

        # randomize camera settings
        dis = np.random.random() * 0.5 + 1
        fov = np.pi / 2 #np.rad2deg(np.arctan2(.5, dis) * 2) * 1.5
        # camera_height = np.random.random() * 1.4
        # if obj_category in ['microwave', 'picture', 'top_cabinet', 'towel_rack', 'wall_clock']:
        #     camera_height -= 1.
        azim = np.random.random() * np.pi * 2
        elev = np.random.random() * np.pi / 3
        cam_pose = np.eye(4)
        y = dis * np.sin(elev)
        x = dis * np.cos(elev) * np.sin(azim)
        z = dis * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        cam = PerspectiveCamera(yfov=fov)
        cam_node = scene.add(cam, pose=cam_pose)
        # direc_l_node = scene.add(spot_l, pose=cam_pose)
        
        color, _ = r.render(scene)
        nm = {node: 255 for _, node in enumerate(scene.mesh_nodes)}
        seg = r.render(scene, RenderFlags.SEG, nm)[0]
        
        bg_node = scene.add(bg_mesh)
        bg_color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES | pyrender.constants.RenderFlags.FLAT)
        mask = seg // 255
        blend = (color * mask + bg_color * (1 - mask)).astype(np.uint8)
        
        seg = np.all(seg, -1).astype(np.uint8)
        bdb2d = seg2obj(seg, 1)['bdb2d']
        for key in ('rgb', 'seg'):
            if key == 'seg' and not args.mask:
                continue
            if key == 'rgb':
                crop = blend[bdb2d['y1']: bdb2d['y2'] + 1, bdb2d['x1']: bdb2d['x2'] + 1]
            else:
                crop = seg[bdb2d['y1']: bdb2d['y2'] + 1, bdb2d['x1']: bdb2d['x2'] + 1]
                crop = crop * 255
            Image.fromarray(crop).save(os.path.join(output_folder, f"render-{i_render:05d}-{key}.png"))
        
        scene.remove_node(bg_node)
        scene.remove_node(cam_node)
        scene.remove_node(spot_l_node)
        scene.remove_node(direc_l_node)

    r.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/rlsd',
                        help='The path of the dataset')
    parser.add_argument('--output', type=str, default='data/rlsd_obj',
                        help='The path of the output folder')
    parser.add_argument('--processes', type=int, default=0,
                        help='Number of threads')
    parser.add_argument('--skip_done', default=False, action='store_true',
                        help='Skip objects exist in output folder')
    parser.add_argument('--object_path', type=str, default=None,
                        help="Specify the 'visual' folder of a single object to be processed")
    parser.add_argument('--renders', type=int, default=10,
                        help='Number of renders per obj')
    parser.add_argument('--shape_id', type=str, default='ZPCD5500')
    # parser.add_argument('--obj_source', type=str, default='wayfair')
    parser.add_argument('--mask', type=bool, default=False)
    args = parser.parse_args()
    
    # render_view(args.out_dir, args.shape_id)
    
    # render and preprocess obj
    # if args.all:
    args_dict = args.__dict__.copy()
    # args_dict['spacing'] = args.bbox / 32
    # args_dict['bbox'] = ' '.join([str(-args.bbox / 2), ] * 3 + [str(args.bbox / 2), ] * 3)
    # print(f"bbox: [{args_dict['bbox']}] spacing: {args_dict['spacing']}")

    if args.object_path is None:
        # object_paths = glob(os.path.join(gibson2.ig_dataset_path, 'objects', '*', '*', '*.urdf'))
        objects = glob('/local-scratch/qiruiw/research/DeepPanoContext/data/rlsd_obj/*/*')
        object_paths = [p.strip() for p in open("/project/3dlg-hcvc/rlsd/data/annotations/unique_shapes.txt")]
        print(f"{len(object_paths)} objects in total")
    else:
        objects = [args.object_path]
        object_paths = [args.object_path]
    obj_path_mapping = {}
    for obj_path in object_paths:
        obj_name = obj_path.split('/')[-1].split('.')[0]
        obj_path_mapping[obj_name] = obj_path
    args_list = []
    for obj in objects:
        obj_name = obj.split('/')[-1].split('.')[0]
        args_dict['object'] = obj
        args_dict['object_path'] = obj_path_mapping[obj_name]
        args_list.append(argparse.Namespace(**args_dict))

    print("Rendering ...")
    if args.processes == 0:
        r = []
        for a in tqdm(args_list):
            r.append(render_view(a))
    else:
        with Pool(processes=args.processes) as p:
            r = list(tqdm(p.imap(render_view, args_list), total=len(args_list)))