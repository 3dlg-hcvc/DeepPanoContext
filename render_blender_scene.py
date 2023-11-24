import os
import glob
import bpy
import json
import numpy as np
import mathutils
import math
import argparse


C = bpy.context
objects = bpy.context.scene.objects
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    if obj.type == "MESH":
        obj.select_set(True)
        bpy.ops.object.delete()
# bpy.ops.object.select_all(action='SELECT')


def _load_mp3d(house_id):
    bpy.ops.object.select_all(action='DESELECT')
    filepath = glob.glob(f"/datasets/internal/matterport/public_extracted/v1/scans/{house_id}/matterport_mesh/*/*.obj")[0]
    bpy.ops.wm.obj_import(filepath=filepath)
    C.object.rotation_euler[0] = 0
    mp3d_name = C.object.name
    # C.object.select_set(False)

    return mp3d_name

def _load_r3ds(task_id, mesh_type):
    bpy.ops.object.select_all(action='DESELECT')
    filepath = f"/project/3dlg-hcvc/rlsd/data/annotations/viz_paper/{mesh_type}/{task_id}/{task_id}.scene/{task_id}.scene.glb"
    assert os.path.exists(filepath), f"{task_id} NOT FOUND!"
    bpy.ops.import_scene.gltf(filepath=filepath)
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
    # scene_bound = np.array([p[:] for p in C.active_object.bound_box])
    # scene_center = scene_bound.mean(0)
    

def _setup_camera(topdown=None, panorama=None, perspective=None, turnaround=None, turntable=None, turn_angle=None, cam2world=None, orthographic=None):
    camera = bpy.data.objects["Camera"]
    bpy.data.cameras["Camera"].type = "PERSP"
    bpy.data.cameras["Camera"].lens_unit = "FOV"
    bpy.data.cameras["Camera"].angle = math.radians(80)
    
    scene_bound = np.array([p[:] for p in C.active_object.bound_box])
    scene_center = scene_bound.mean(0)
    radius = np.linalg.norm(scene_bound - scene_center, axis=1).max()
    print(radius)
    
    if topdown:
        fov = np.arctan2(radius, 8) * 2
        bpy.data.cameras["Camera"].angle = fov
        # bpy.data.cameras["Camera"].type = "ORTHO"
        # bpy.data.cameras["Camera"].ortho_scale = 7.31429 * radius / 3.68
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
        bpy.context.scene.render.resolution_x = 960
        bpy.context.scene.render.resolution_y = 540
        
    if turntable:
        assert turn_angle is not None
        if orthographic:
            bpy.data.cameras["Camera"].type = "ORTHO"
            bpy.data.cameras["Camera"].ortho_scale = 7.31429 * radius / 3.68
        else:
            fov = np.arctan2(radius, 8) * 2
            bpy.data.cameras["Camera"].angle = fov
        camera_pos = np.array(C.active_object.location) + np.array([0, 0, 8])
        r = 8 / np.sqrt(3)
        angle = 2*np.pi * ((turn_angle - 90) / 360)
        camera_pos[:2] += np.array([r*np.cos(angle), r*np.sin(angle)])
        view_dir = mathutils.Vector(np.array(C.active_object.location) - camera_pos)
        camera_rot_quat = view_dir.to_track_quat('-Z', 'Y')
        camera_rot = np.array(camera_rot_quat.to_euler())
        bpy.context.scene.render.resolution_x = 500
        bpy.context.scene.render.resolution_y = 500

    # Add a camera, pointing at the center of the object
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
    enode.image = bpy.data.images.load("/project/3dlg-hcvc/rlsd/data/annotations/viz_paper/photo_studio_01_2k.hdr")
    bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].projection = 'EQUIRECTANGULAR'
    C.scene.world.node_tree.links.new(enode.outputs['Color'], C.scene.world.node_tree.nodes['Background'].inputs['Color'])

    if use_cycles:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.adaptive_threshold = 0.1
        bpy.context.scene.cycles.samples = 1024
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = 64


def _render(filepath):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    
def _clean():
    # Clear the scene
    for obj in bpy.context.scene.objects:
        if obj.type not in ["CAMERA", "LIGHT"]:
            obj.select_set(True)
    # bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # # Clear the materials
    # for material in bpy.data.materials:
    #     material.user_clear()
    #     bpy.data.materials.remove(material)


def render(full_task_id, mesh_type, topdown=None, perspective=None, turnaround=None, turntable=None, render_mp3d=None, orthographic=None):
    # task_pano_mapping = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/task_pano_mapping.json"))
    # full_pano_id = task_pano_mapping[task_id]
    house_id, level_id, pano_id, task_id = full_task_id.split("_")
    
    out_dir = f"/project/3dlg-hcvc/rlsd/data/annotations/viz_paper/exported_glb_renders/{mesh_type}/{task_id}"
    # if os.path.exists(out_dir):
    #     return
    os.makedirs(out_dir, exist_ok=True)

    with open(f"/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_camera_poses/{house_id}.jsonl") as f:
        cameras = f.readlines()
        for c in cameras:
            c = json.loads(c)
            if c["id"] == pano_id:
                camera = c
                break
    cam2world = np.array(camera["camera"]["extrinsics"]).reshape(4, 4)

    _load_r3ds(task_id, mesh_type)
    _init()

    if topdown:
        _setup_camera(topdown==True)
        _render(os.path.join(out_dir, "topdown.png"))

    if perspective:
        _setup_camera(perspective=True, turn_angle=45, cam2world=cam2world)
        _render(os.path.join(out_dir, "perspective_sample.png"))
            
    if turntable:
        tt_out_dir = os.path.join(out_dir, "turntable")
        if orthographic:
            tt_out_dir += '_orth'
        os.makedirs(tt_out_dir, exist_ok=True)
        for tt_idx in range(360):
            _setup_camera(turntable=True, turn_angle=tt_idx, orthographic=orthographic)
            _render(os.path.join(tt_out_dir, f"{tt_idx}.png"))

    if turnaround:
        tt_out_dir = os.path.join(out_dir, "turnaround")
        os.makedirs(tt_out_dir, exist_ok=True)
        for tt_idx in range(360):
            _setup_camera(perspective=True, turnaround=True, turn_angle=tt_idx, cam2world=cam2world)
            _render(os.path.join(tt_out_dir, f"{tt_idx}.png"))
        
        if render_mp3d:
            C.active_object.hide_render = True
            _load_mp3d(house_id)
            tt_out_dir = os.path.join(out_dir, "turnaround_mp3d")
            os.makedirs(tt_out_dir, exist_ok=True)
            for tt_idx in range(360):
                _setup_camera(perspective=True, turnaround=True, turn_angle=tt_idx, cam2world=cam2world)
                _render(os.path.join(tt_out_dir, f"{tt_idx}.png"))
            
    _clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Blender render.')
    parser.add_argument('--tasks', default="/project/3dlg-hcvc/rlsd/data/annotations/viz_paper/selected_task_pano_ids.txt", type=str)
    parser.add_argument('--mesh_type', required=True, type=str)
    parser.add_argument('--render_mp3d', default=False, action='store_true')
    parser.add_argument('--topdown', default=False, action='store_true')
    parser.add_argument('--perspective', default=False, action='store_true')
    parser.add_argument('--turntable', default=False, action='store_true')
    parser.add_argument('--turnaround', default=False, action='store_true')
    parser.add_argument('--orth', default=False, action='store_true')
    args = parser.parse_args()

    # render("61e0e083ddd48e322a187d89", topdown=True, perspective=True, turnaround=True)
    # render("PX4nDJXEHrG_L1_9a65ec8ec80d41a492cf617c83b15ec6_6420eb64be0192bc6a4dad5f", "exported_glb_by_instance", topdown=True)
    # render("8WUmhLawc2A_L0_0fd8c430bdb34aedb5c11b56ceb13b63_61e0e084ddd48e322a187e9d", "exported_glb_by_instance", turntable=True, orthographic=True)
    # render("61e0e083ddd48e322a187d89", turntable=True)
    
    scenes = [s.strip() for s in open(args.tasks)]
    
    # for mesh_type in ["exported_glb_by_instance", "exported_glb_by_semantic", "exported_glb_by_modelId", "exported_glb_w_texture"]:
    for scene in scenes:
        house_id, level_id, pano_id, task_id = scene.split("_")
        file_path = f"/project/3dlg-hcvc/rlsd/data/annotations/viz_paper/{args.mesh_type}/{task_id}/{task_id}.scene/{task_id}.scene.glb"
        if not os.path.exists(file_path):
            continue
        render(scene, args.mesh_type, topdown=args.topdown, perspective=args.perspective, turntable=args.turntable, turnaround=args.turnaround, render_mp3d=args.render_mp3d, orthographic=args.orth)