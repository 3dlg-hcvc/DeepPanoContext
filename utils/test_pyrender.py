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


scene_mesh = trimesh.load("/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25/2t7WUuJeko7_L0/1a0d730696cc4057b0037a75c8ef6b59/631fb1a2fa528b15f94d04e2/scene_mesh.obj")
center = np.mean(scene_mesh.bounds, axis=0)
radius = np.linalg.norm(scene_mesh.vertices - center, axis=1).max()
center_mat = np.array([
[1, 0, 0, -center[0]],
[0.0, 1, 0.0, -center[1]],
[0.0, 0.0, 1, -center[2]],
[0.0,  0.0, 0.0, 1.0]
])
scene_mesh.apply_transform(center_mat)

mat = pyrender.MetallicRoughnessMaterial(alphaMode="BLEND", baseColorFactor=(162/255, 191/255, 254/255, 0.6))
pyr_mesh = pyrender.Mesh.from_trimesh(scene_mesh, material=mat)

bdb3d_mesh = trimesh.load("/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25/2t7WUuJeko7_L0/1a0d730696cc4057b0037a75c8ef6b59/631fb1a2fa528b15f94d04e2/layout_bdb3d.obj")
bdb3d_mesh.apply_transform(center_mat)
pyr_bdb3d_mesh = pyrender.Mesh.from_trimesh(bdb3d_mesh)

if not isinstance(scene_mesh, trimesh.Scene):
    scene = trimesh.Scene(scene_mesh)
# scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.1, 0.1, 0.1])

scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1, 1.0])
scene.add(pyr_mesh)
scene.add(pyr_bdb3d_mesh)

# scene.add(mesh)
# pyrender.Viewer(scene)

r = OffscreenRenderer(viewport_width=1024, viewport_height=1024)
direc_l = DirectionalLight(color=np.ones(3), intensity=2.5)
# randomize camera settings
dis = 6
fov = np.arctan2(4, dis) * 2
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
# spot_l_node = scene.add(spot_l)
# spot_l_node = scene.add(spot_l, pose=cam_pose)

color, _ = r.render(scene)
Image.fromarray(color).save("111.png")