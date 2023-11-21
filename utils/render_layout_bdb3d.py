import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import trimesh
import pyrender
from pyrender import RenderFlags
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer


def render_view(in_file, out_file, gt_file=None):
    # output_folder = os.path.join(args.output, *args.object.split('/')[-2:])
    # os.makedirs(output_folder, exist_ok=True)
    # if args.skip_done and len(glob(os.path.join(output_folder, f"render-*.png"))) == args.renders:
    #     return

    scene_mesh = trimesh.load(in_file)

    center = np.mean(scene_mesh.bounds, axis=0)
    radius = np.linalg.norm(scene_mesh.vertices - center, axis=1).max()
    # scale = 1.0 / float(max(obj.bounds[1] - obj.bounds[0]))
    center_mat = np.array([
    [1, 0, 0, -center[0]],
    [0.0, 1, 0.0, -center[1]],
    [0.0, 0.0, 1, -center[2]],
    [0.0,  0.0, 0.0, 1.0]
    ])
    # norm_mat = np.array([
    #     [scale, 0, 0, 0],
    #     [0.0, scale, 0.0, 0],
    #     [0.0, 0.0, scale, 0],
    #     [0.0,  0.0, 0.0, 1.0]
    # ])
    # obj.apply_transform(np.matmul(norm_mat, center_mat))
    scene_mesh.apply_transform(center_mat)
    if not isinstance(scene_mesh, trimesh.Scene):
        scene = trimesh.Scene(scene_mesh)
    scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.2, 0.2, 0.2])
    
    if gt_file is not None:
        gt_mesh = trimesh.load(gt_file)
        gt_mesh.apply_transform(center_mat)
        mat = pyrender.MetallicRoughnessMaterial(alphaMode="BLEND", baseColorFactor=(117/255, 187/255, 253/255, 0.6))
        gt_mesh = pyrender.Mesh.from_trimesh(gt_mesh, material=mat)
        scene.add(gt_mesh)

    direc_l = DirectionalLight(color=np.ones(3), intensity=2.5)
    # spot_l = SpotLight(color=np.ones(3), intensity=1.0, innerConeAngle=np.pi/8, outerConeAngle=np.pi/3)
    # point_l = PointLight(color=np.ones(3), intensity=10.0)
    r = OffscreenRenderer(viewport_width=1024, viewport_height=1024)

    # randomize camera settings
    dis = 8
    fov = np.arctan2(radius+2, dis) * 2
    # fov = np.pi / 2 #np.rad2deg(np.arctan2(.5, dis) * 2) * 1.5
    # camera_height = np.random.random() * 1.4
    azim = 0 # - np.pi / 5 #
    elev = - np.pi / 6 #
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
    # roty = np.array([
    #     [np.cos(azim), 0, np.sin(azim)],
    #     [0.0, 1, 0.0],
    #     [-np.sin(azim), 0, np.cos(azim)]
    # ])
    rotz = np.array([
        [np.cos(azim), -np.sin(azim), 0.],
        [np.sin(azim), np.cos(azim), 0.0],
        [0., 0., 1.]
    ])
    cam_pose[:3, :3] = np.matmul(rotz, rotx)
    cam = PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    cam_node = scene.add(cam, pose=cam_pose)
    direc_l_node = scene.add(direc_l)
    # direc_l_node = scene.add(direc_l, pose=cam_pose)

    
    color, _ = r.render(scene)
    Image.fromarray(color).save(out_file)
    
    scene.remove_node(cam_node)
    # scene.remove_node(direc_l_node)

    r.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/rlsd',
                        help='The path of the dataset')
    parser.add_argument('--output', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/rlsd_obj',
                        help='The path of the output folder')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip rendering')
    parser.add_argument('--skip_done', default=False, action='store_true',
                        help='Skip objects exist in output folder')
    parser.add_argument('--object_path', type=str, default=None,
                        help="Specify the 'visual' folder of a single object to be processed")
    parser.add_argument('--renders', type=int, default=1,
                        help='Number of renders per obj')
    parser.add_argument('--max_renders_per_obj', type=int, default=25,
                        help='Number of renders per obj selected for splits')
    parser.add_argument('--shape_id', type=str, default='ZPCD5500')
    # parser.add_argument('--obj_source', type=str, default='wayfair')
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--train', type=float, default=0.9,
                        help='Ratio of train split')
    args = parser.parse_args()
    
    render_view("/local-scratch/qiruiw/research/DeepPanoContext/mesh.ply")
    
    