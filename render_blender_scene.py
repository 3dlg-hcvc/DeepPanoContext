import bpy
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

bpy.ops.import_scene.gltf(filepath="/project/3dlg-hcvc/rlsd/data/annotations/exported_scene_assets/61e0e083ddd48e322a187d89.scene/61e0e083ddd48e322a187d89.scene.glb")

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
    
    
camera = bpy.data.objects["Camera"]
camera_pos = np.array(C.active_object.location) + np.array([0, 0, 8])

# Calculate rotation
view_dir = mathutils.Vector(np.array([0, 0, -1]))
camera_rot_quat = view_dir.to_track_quat('-Z', 'Y')
camera_rot = np.array(camera_rot_quat.to_euler())

# Add a camera, pointing at the center of the object
bpy.data.cameras["Camera"].lens_unit = "FOV"
bpy.data.cameras["Camera"].angle = math.radians(75)
camera.location = camera_pos
camera.rotation_euler = camera_rot


# Set render resolution
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

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

# bpy.ops.image.open(filepath="/local-scratch/qiruiw/research/DeepPanoContext/data/background/photo_studio_01_2k.hdr", directory="/local-scratch/qiruiw/research/DeepPanoContext/data/background", files=[{"name":"photo_studio_01_2k.hdr", "name":"photo_studio_01_2k.hdr"}], relative_path=True, show_multiview=False)

C.scene.world.use_nodes = True
# bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image
enode = C.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
enode.image = bpy.data.images.load("/local-scratch/qiruiw/research/DeepPanoContext/data/background/photo_studio_01_2k.hdr")
bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].projection = 'EQUIRECTANGULAR'
C.scene.world.node_tree.links.new(enode.outputs['Color'], C.scene.world.node_tree.nodes['Background'].inputs['Color'])


bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.adaptive_threshold = 0.1
bpy.context.scene.cycles.samples = 1024

bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = "test_blender.png"
bpy.ops.render.render(write_still=True)