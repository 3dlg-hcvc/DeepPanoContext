import os
import bpy
import json
import numpy as np
import mathutils
import math


C = bpy.context
objects = bpy.context.scene.objects
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    if obj.type == "MESH":
        obj.select_set(True)
        bpy.ops.object.delete()
# bpy.ops.object.select_all(action='SELECT')


def _load_scene(task_id):
    bpy.ops.import_scene.gltf(filepath=f"/project/3dlg-hcvc/rlsd/data/annotations/exported_scene_assets/{task_id}.scene/{task_id}.scene.glb")
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    r3ds_meshes = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            r3ds_meshes.append(obj)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        else:
            obj.select_set(False)

    # with C.temp_override(active_object=C.active_object, selected_editable_objects=r3ds_meshes):
    bpy.ops.object.join()
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    scene_bound = np.array([p[:] for p in C.active_object.bound_box])
    scene_center = scene_bound.mean(0)
    

def _setup_camera(topdown=None, panorama=None, perspective=None, turnaround=None, turntable=None, turn_angle=None, cam2world=None):
    camera = bpy.data.objects["Camera"]
    
    if topdown:
        camera_pos = np.array(C.active_object.location) + np.array([0, 0, 8])
        view_dir = mathutils.Vector(np.array([0, 0, -1]))
        camera_rot_quat = view_dir.to_track_quat('-Z', 'Y')
        camera_rot = np.array(camera_rot_quat.to_euler())
    
    if panorama:
        assert cam2world is not None
        align = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        recover = np.array([[math.cos(np.pi/2), 0, math.sin(np.pi/2)],
           [0, 1, 0],
           [-math.sin(np.pi/2), 0, math.cos(np.pi/2)]])
        cam2world[:3, :3] = cam2world[:3, :3] @ recover @ align
        bpy.data.objects["Camera"].matrix_world = mathutils.Matrix(cam2world)

    if perspective:
        assert cam2world is not None
        camera_pos = cam2world[:3, 3]
        view_dir = mathutils.Vector(np.array([0, 1, -0.2]))
        camera_rot_quat = view_dir.to_track_quat('-Z', 'Y')
        camera_rot = np.array(camera_rot_quat.to_euler())

    if turnaround:
        assert turn_angle is not None
        camera_rot[2] += 2*np.pi / 360 * turn_angle
        
    if turntable:
        assert turn_angle is not None
        camera_pos = np.array(C.active_object.location) + np.array([0, 0, 8])
        r = 8 / np.sqrt(3)
        angle = 2*np.pi * ((turn_angle - 90) / 360)
        camera_pos[:2] += np.array([r*np.cos(angle), r*np.sin(angle)])
        view_dir = mathutils.Vector(np.array(C.active_object.location) - camera_pos)
        camera_rot_quat = view_dir.to_track_quat('-Z', 'Y')
        camera_rot = np.array(camera_rot_quat.to_euler())

    # Add a camera, pointing at the center of the object
    bpy.data.cameras["Camera"].lens_unit = "FOV"
    bpy.data.cameras["Camera"].angle = math.radians(75)
    camera.location = camera_pos
    camera.rotation_euler = camera_rot


def _init(use_cycles=None):
    # Set render resolution
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024

    # Shadow settings
    bpy.context.scene.eevee.shadow_cube_size = "1024"
    bpy.context.scene.eevee.use_shadow_high_bitdepth = True
    bpy.context.scene.eevee.use_soft_shadows = True

    # Set skybox to transparent
    bpy.context.scene.render.film_transparent = True

    bpy.context.scene.view_settings.exposure = -1.4

    light = bpy.data.objects["Light"]
    bpy.data.lights["Light"].energy = 2000
    bpy.data.lights["Light"].shadow_soft_size = 0.2
    light.location = np.array(C.active_object.location) + np.array([-1, -1, 6])

    C.scene.world.use_nodes = True
    # bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image
    enode = C.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    enode.image = bpy.data.images.load("/local-scratch/qiruiw/research/DeepPanoContext/data/background/photo_studio_01_2k.hdr")
    bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].projection = 'EQUIRECTANGULAR'
    C.scene.world.node_tree.links.new(enode.outputs['Color'], C.scene.world.node_tree.nodes['Background'].inputs['Color'])

    if use_cycles:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.adaptive_threshold = 0.1
        bpy.context.scene.cycles.samples = 1024
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = 100


def _render(filepath):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def render(task_id, topdown=None, perspective=None, turnaround=None, turntable=None):
    out_dir = f"/project/3dlg-hcvc/rlsd/data/annotations/exported_scene_renders/{task_id}"
    os.makedirs(out_dir, exist_ok=True)

    task_pano_mapping = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/task_pano_mapping.json"))
    full_pano_id = task_pano_mapping[task_id]
    house_id, level_id, pano_id = full_pano_id.split("_")

    with open(f"/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_camera_poses/{house_id}.jsonl") as f:
        cameras = f.readlines()
        for c in cameras:
            c = json.loads(c)
            if c["id"] == pano_id:
                camera = c
                break
    cam2world = np.array(camera["camera"]["extrinsics"]).reshape(4, 4)

    _load_scene(task_id)
    _init()

    if topdown:
        _setup_camera(topdown==True)
        _render(os.path.join(out_dir, "topdown.png"))

    if perspective:
        _setup_camera(perspective=True, turntable_idx=45, cam2world=cam2world)
        _render(os.path.join(out_dir, "perspective_sample.png"))

    if turnaround:
        tt_out_dir = os.path.join(out_dir, "turnaround")
        os.makedirs(tt_out_dir, exist_ok=True)
        for tt_idx in range(360):
            _setup_camera(perspective=True, turnaround=True, turn_angle=tt_idx, cam2world=cam2world)
            _render(os.path.join(tt_out_dir, f"{tt_idx}.png"))
            
    if turntable:
        tt_out_dir = os.path.join(out_dir, "turntable")
        os.makedirs(tt_out_dir, exist_ok=True)
        for tt_idx in range(360):
            _setup_camera(perspective=True, turntable=True, turn_angle=tt_idx, cam2world=cam2world)
            _render(os.path.join(tt_out_dir, f"{tt_idx}.png"))


# render("61e0e083ddd48e322a187d89", topdown=True, perspective=True, turnaround=True)
render("61e0e083ddd48e322a187d89", turntable=True)