import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import sys
import subprocess
import argparse
import trimesh

from utils.igibson_utils import IGScene, hash_split
from utils.image_utils import save_image
from models.pano3d.dataloader import IGSceneDataset
import numpy as np
from scipy.spatial import cKDTree

# import gibson2
# from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
# from gibson2.simulator import Simulator
# from gibson2.objects.articulated_object import ArticulatedObject

from utils.render_utils import seg2obj, render_camera, hdr_texture, hdr_texture2
from .mesh_utils import MeshIO, save_mesh, load_mesh, normalize_to_unit_square, read_obj, write_obj, \
    sample_pnts_from_obj
from utils.basic_utils import read_json, write_json


def remove_if_exists(f):
    if os.path.exists(f):
        os.remove(f)


def normalize(input_path, output_folder):
    output_path = os.path.join(output_folder, 'mesh_normalized.obj')

    obj_data = read_obj(input_path, ['v', 'f'])
    obj_data['v'] = normalize_to_unit_square(obj_data['v'])[0]
    write_obj(output_path, obj_data)
    return output_path


def process_mgnet(obj_path, output_folder):
    obj_data = read_obj(obj_path, ['v', 'f'])
    sampled_points = sample_pnts_from_obj(obj_data, 10000, mode='random')
    sampled_points.tofile(os.path.join(output_folder, f'gt_3dpoints.mgn'))

    tree = cKDTree(sampled_points)
    dists, indices = tree.query(sampled_points, k=30)
    densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices])
    densities.tofile(os.path.join(output_folder, f'densities.mgn'))


def make_watertight(input_path, output_folder):
    mesh_fusion = 'external/mesh_fusion'
    output_path = os.path.join(output_folder, 'mesh_watertight.obj')

    # convert mesh to off
    off_path = os.path.splitext(output_path)[0] + '.off'
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {input_path} -o {off_path}',
                            shell=True)

    # scale mesh
    subprocess.check_output(f'{sys.executable} {mesh_fusion}/scale.py'
                            f' --in_file {off_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite',
                            shell=True)

    # create depth maps
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" {sys.executable} {mesh_fusion}/fusion.py'
                            f' --mode=render --in_file {off_path} --out_dir {output_folder} --overwrite',
                            shell=True)

    # produce watertight mesh
    depth_path = off_path + '.h5'
    transform_path = os.path.splitext(output_path)[0] + '.npz'
    subprocess.check_output(f'{sys.executable} {mesh_fusion}/fusion.py --mode=fuse'
                            f' --in_file {depth_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite',
                            shell=True)

    # remove isolated meshes
    obj_path = os.path.splitext(output_path)[0] + '.obj'
    watertight_trimesh = load_mesh(obj_path)
    connected_components = trimesh.graph.split(watertight_trimesh, only_watertight=True)
    volumes = [m.volume for m in connected_components]
    watertight_trimesh = connected_components[np.argmax(volumes)]
    save_mesh(watertight_trimesh, obj_path)

    # simplify mesh
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" {sys.executable} {mesh_fusion}/simplify.py'
                            f' --in_file={obj_path} --out_dir {output_folder}', shell=True)

    os.remove(off_path)
    os.remove(transform_path)
    os.remove(depth_path)
    return output_path


def crop_images(args):
    scene = IGScene.from_pickle(args.camera)
    crop_types = ('rgb', 'seg') if args.mask else ('rgb',)
    scene.crop_images(perspective=True, short_width=args.crop_width, crop_types=crop_types)
    for obj in scene['objs']:
        if args.object_path is not None:
            model_path = os.path.join(*args.object_path.split('/')[-4:-2])
            if model_path != obj['model_path']:
                continue
        output_folder = os.path.join(args.output, obj["classname"][0], obj["model_name"].split('.')[-1])
        for key in crop_types:
            crop = obj[key]
            ext = '' if key == 'rgb' else f"-{key}"
            save_image(crop, os.path.join(output_folder, f"crop-{scene['scene'].replace('/', '-')}-{scene['name']}-{obj['id']}{ext}.png"))


def preprocess_obj(args):
    gaps = './external/ldif/gaps/bin/x86_64'
    output_folder = os.path.join(args.output, *args.object.split('/')[-2:])
    obj_category = output_folder.split('/')[-2]
    # output_folder = './demo_obj'
    os.makedirs(output_folder, exist_ok=True)
    if args.skip_done and os.path.exists(os.path.join(output_folder, 'uniform_points.sdf')):
        return

    if not args.skip_render or not args.skip_watertight:
        # merge obj files and estimate scale
        # obj_list = glob(os.path.join(os.path.dirname(args.object_path), 'shape', 'visual', '*.obj'))
        # merged_mesh = MeshIO.from_file(obj_list).load().merge()
        merged_mesh = MeshIO.from_file([args.object_path]).load().merge()
        v = np.array(merged_mesh.vertices, dtype=np.float32)
        obj_bbox = np.max(v, axis=0) - np.min(v, axis=0)
        scale = 1. / float(max(obj_bbox))
        center = np.mean(merged_mesh.bounds, axis=0)
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
        merged_mesh.apply_transform(np.matmul(norm_mat, center_mat))

        # # set renderer
        # settings = MeshRendererSettings(
        #     env_texture_filename=hdr_texture,
        #     env_texture_filename2=hdr_texture2,
        #     env_texture_filename3=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'Rs.hdr'),
        #     enable_shadow = True, msaa = True, enable_pbr=True
        # )
        # s = Simulator(mode='headless', image_width=512, image_height=512,
        #               device_idx=args.gpu_id, rendering_settings=settings)
        # renderer = s.renderer

        # load object
        # obj = ArticulatedObject(filename=args.object_path, scale=scale)
        # s.import_object(obj, class_id=1)
        # instance = renderer.instances[0]
        # merged_mesh = [trimesh.Trimesh(v, f) for v, f in zip(*instance.dump())]
        # merged_mesh = sum(merged_mesh)

    # # render with pyrender
    # if not args.skip_render:
    #     obj = trimesh.load_mesh(args.object_path)
    #     center = np.mean(obj.bounds, axis=0)
    #     scale = 1.0 / np.linalg.norm(obj.bounds[1] - obj.bounds[0])
    #     center_mat = np.array([
    #     [1, 0, 0, -center[0]],
    #     [0.0, 1, 0.0, -center[1]],
    #     [0.0, 0.0, 1, -center[2]],
    #     [0.0,  0.0, 0.0, 1.0]
    #     ])
    #     norm_mat = np.array([
    #         [scale, 0, 0, 0],
    #         [0.0, scale, 0.0, 0],
    #         [0.0, 0.0, scale, 0],
    #         [0.0,  0.0, 0.0, 1.0]
    #     ])
    #     obj.apply_transform(np.matmul(norm_mat, center_mat))
    #     scene = pyrender.Scene(ambient_light=[1, 1, 1])
    #     obj_node = scene.add(pyrender.Mesh.from_trimesh(obj))
        
    #     # scene = pyrender.Scene.from_trimesh_scene(obj, ambient_light=[0.2, 0.2, 0.2])
    #     # scene = pyrender.Scene.from_trimesh_scene(obj)
        
    #     img = np.asarray(Image.open("/local-scratch/qiruiw/research/DeepPanoContext/ballroom_2k.png"))
    #     sphere_trimesh = trimesh.load_mesh("/local-scratch/qiruiw/research/rlsd/evaluation/conf/sphere/sphere.obj")
    #     sphere_scale = np.array([
    #         [12, 0, 0, 0],
    #         [0.0, 12, 0.0, 0],
    #         [0.0, 0.0, 12, 0],
    #         [0.0,  0.0, 0.0, 1.0]
    #     ])
    #     sphere_trimesh.apply_transform(sphere_scale)
    #     texture = pyrender.Texture(source=img, source_channels="RGB", 
    #                             sampler=pyrender.Sampler(magFilter=pyrender.constants.GLTF.NEAREST, 
    #                                                         minFilter=pyrender.constants.GLTF.NEAREST))
    #     material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.,1.,1.,1.],baseColorTexture=texture)
    #     bg_mesh = pyrender.Mesh.from_trimesh(sphere_trimesh, material=material)
        
    #     # direc_l = DirectionalLight(color=np.ones(3), intensity=0.5)
    #     spot_l = SpotLight(color=np.ones(3), intensity=1.0, innerConeAngle=np.pi/8, outerConeAngle=np.pi/3)
    #     r = OffscreenRenderer(viewport_width=512, viewport_height=512)
        
    #     for i_render in range(args.renders):
    #         # randomize light direction
    #         # renderer.set_light_position_direction(((np.random.random(3) - 0.5) * 10 + 5).tolist(), [0, 0, 0])
    #         # l_dis = np.random.random() * 5 + 2
    #         # azim = np.random.random() * np.pi * 2
    #         # elev = np.random.random() * np.pi
    #         # l_pose = np.eye(4)
    #         # y = l_dis * np.sin(elev)
    #         # x = l_dis * np.cos(elev) * np.sin(azim)
    #         # z = l_dis * np.cos(elev) * np.cos(azim)
    #         # l_pose[:3, 3] = [x, y, z]
    #         # rotx = np.array([
    #         #     [1.0, 0, 0],
    #         #     [0.0, np.cos(elev), np.sin(elev)],
    #         #     [0.0, -np.sin(elev), np.cos(elev)]
    #         # ])
    #         # roty = np.array([
    #         #     [np.cos(azim), 0, np.sin(azim)],
    #         #     [0.0, 1, 0.0],
    #         #     [-np.sin(azim), 0, np.cos(azim)]
    #         # ])
    #         # l_pose[:3, :3] = np.matmul(roty, rotx)
    #         # direc_l_node = scene.add(spot_l, pose=cam_pose)

    #         # randomize camera settings
    #         dis = 1 #np.random.random() * 1 + 2
    #         fov = (np.pi / 2.0) #np.rad2deg(np.arctan2(.5, dis) * 2) * 1.5
    #         # camera_height = np.random.random() * 1.4
    #         # if obj_category in ['microwave', 'picture', 'top_cabinet', 'towel_rack', 'wall_clock']:
    #         #     camera_height -= 1.
    #         azim = np.random.random() * np.pi * 2
    #         elev = np.random.random() * np.pi / 3
    #         cam_pose = np.eye(4)
    #         y = dis * np.sin(elev)
    #         x = dis * np.cos(elev) * np.sin(azim)
    #         z = dis * np.cos(elev) * np.cos(azim)
    #         cam_pose[:3, 3] = [x, y, z]
    #         rotx = np.array([
    #             [1.0, 0, 0],
    #             [0.0, np.cos(elev), np.sin(elev)],
    #             [0.0, -np.sin(elev), np.cos(elev)]
    #         ])
    #         roty = np.array([
    #             [np.cos(azim), 0, np.sin(azim)],
    #             [0.0, 1, 0.0],
    #             [-np.sin(azim), 0, np.cos(azim)]
    #         ])
    #         cam_pose[:3, :3] = np.matmul(roty, rotx)
    #         cam = PerspectiveCamera(yfov=fov)
    #         cam_node = scene.add(cam, pose=cam_pose)
    #         direc_l_node = scene.add(spot_l, pose=cam_pose)
            
    #         # pos = np.array([np.sin(yaw) * dis, np.cos(yaw) * dis, camera_height], np.float32)
    #         # target = np.array([0, 0, 0], dtype= np.float32)
    #         # camera = {
    #         #     'pos': pos, 'target': target, 'up': np.array([0, 0, 1], np.float32),
    #         #     'width': 512, 'height': 512, 'vertical_fov': fov
    #         # }

    #         # render
    #         # render_results = render_camera(renderer, camera, ['rgb', 'seg'], perspective=True)
    #         # visualize = visualize_image(render_results)
    #         # show_image(visualize['rgb'])
    #         # show_image(visualize['seg'])
            
    #         color, _ = r.render(scene)
    #         nm = {node: 255 for _, node in enumerate(scene.mesh_nodes)}
    #         seg = r.render(scene, RenderFlags.SEG, nm)[0]
            
    #         bg_node = scene.add(bg_mesh)
    #         bg_color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES | pyrender.constants.RenderFlags.FLAT)
    #         mask = seg // 255
    #         blend = (color * mask + bg_color * (1 - mask)).astype(np.uint8)
            
    #         seg = np.all(seg, -1).astype(np.uint8)
    #         bdb2d = seg2obj(seg, 1)['bdb2d']
    #         for key in ('rgb', 'seg'):
    #             if key == 'seg' and not args.mask:
    #                 continue
    #             if key == 'rgb':
    #                 crop = blend[bdb2d['y1']: bdb2d['y2'] + 1, bdb2d['x1']: bdb2d['x2'] + 1]
    #             else:
    #                 crop = seg[bdb2d['y1']: bdb2d['y2'] + 1, bdb2d['x1']: bdb2d['x2'] + 1]
    #                 crop = crop * 255
    #             save_image(crop, os.path.join(output_folder, f"render-{i_render:05d}-{key}.png"))

    #         # # crop and resize image
    #         # bdb2d = seg2obj(render_results['seg'], 1)['bdb2d']
    #         # for key in ('rgb', 'seg'):
    #         #     if key == 'seg' and not args.mask:
    #         #         continue
    #         #     crop = render_results[key][bdb2d['y1']: bdb2d['y2'] + 1, bdb2d['x1']: bdb2d['x2'] + 1]
    #         #     # show_image(crop)
    #         #     if key == 'seg':
    #         #         crop = crop * 255
    #         #     save_image(crop, os.path.join(output_folder, f"render-{i_render:05d}-{key}.png"))
            
    #         scene.remove_node(bg_node)
    #         scene.remove_node(cam_node)
    #         scene.remove_node(direc_l_node)

    if not args.skip_watertight:
        # save merged obj
        merged_obj = os.path.join(output_folder, "mesh_merged.obj")
        save_mesh(merged_mesh, merged_obj)

        # Step 0) Normalize and watertight the mesh before applying all other operations.
        normalized_obj = normalize(merged_obj, output_folder)
        make_watertight(normalized_obj, output_folder)

        if not args.keep_interfile:
            remove_if_exists(merged_obj)

    # if not args.skip_render or not args.skip_watertight:
    #     s.disconnect()

    watertight_obj = os.path.join(output_folder, 'mesh_watertight.obj')

    if not args.skip_ldif:
        # convert mesh to ply
        normalized_ply = os.path.splitext(normalized_obj)[0] + '.ply'
        subprocess.check_output(
            f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {normalized_obj} -o {normalized_ply}',
            shell=True)
        watertight_ply = os.path.splitext(watertight_obj)[0] + '.ply'
        subprocess.check_output(
            f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {watertight_obj} -o {watertight_ply}',
            shell=True)

        scaled_ply = os.path.join(output_folder, 'mesh_scaled.ply')
        os.system(f'{gaps}/msh2msh {watertight_ply} {scaled_ply} -scale_by_pca -translate_by_centroid'
                  f' -scale {0.25} -debug_matrix {output_folder}/orig_to_gaps.txt')

        # Step 1) Generate the coarse inside/outside grid:
        os.system(f'{gaps}/msh2df {scaled_ply} {output_folder}/coarse_grid.grd'
                  f' -bbox {args.bbox} -border 0 -spacing {args.spacing} -estimate_sign')

        # Step 2) Generate the near surface points:
        os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/nss_points.sdf'
                  f' -near_surface -max_distance {args.spacing} -num_points 100000 -binary_sdf')

        # Step 3) Generate the uniform points:
        os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/uniform_points.sdf'
                  f' -uniform_in_bbox -bbox {args.bbox} -npoints 100000 -binary_sdf')

        if not args.keep_interfile:
            remove_if_exists(normalized_obj)
            remove_if_exists(watertight_obj)
            remove_if_exists(normalized_ply)
            remove_if_exists(scaled_ply)

    if not args.skip_mgn:
        process_mgnet(watertight_obj, output_folder)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess rlsd objects for single image reconstruction network training.')
    parser.add_argument('--dataset', type=str, default='data/rlsd',
                        help='The path of the dataset')
    parser.add_argument('--output', type=str, default='data/rlsd_obj',
                        help='The path of the output folder')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--keep_interfile', default=False, action='store_true',
                        help='Keep intermediate files')
    parser.add_argument('--skip_done', default=False, action='store_true',
                        help='Skip objects exist in output folder')
    parser.add_argument('--bbox', type=float, default=1.4,
                        help='Width of the bounding box for LDIF in/out grid and uniform sampling')
    parser.add_argument('--object_path', type=str, default=None,
                        help="Specify the 'visual' folder of a single object to be processed")
    parser.add_argument('--renders', type=int, default=100,
                        help='Number of renders per obj')
    parser.add_argument('--skip_watertight', default=False, action='store_true',
                        help='Skip watertight processing')
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip rendering')
    parser.add_argument('--skip_crop', default=False, action='store_true',
                        help='Skip cropping from scene rendering')
    parser.add_argument('--skip_ldif', default=False, action='store_true',
                        help='Skip generating dataset for LDIF')
    parser.add_argument('--skip_mgn', default=False, action='store_true',
                        help='Skip generating dataset for MGNet')
    parser.add_argument('--train', type=float, default=0.9,
                        help='Ratio of train split')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--crop_width', type=int, default=280,
                        help='Shortest edge of images cropped from scene rendering')
    parser.add_argument('--split_by_obj', default=False, action='store_true',
                        help='Split train/test set by object instead of scene')
    parser.add_argument('--split_by_image', default=False, action='store_true',
                        help='Split train/test set by image instead of scene')
    parser.add_argument('--mask', default=False, action='store_true',
                        help='Crop masks with RGBs')
    args = parser.parse_args()

    # crop images
    if not args.skip_crop:
        # delete old crops
        old_crops = glob(os.path.join(args.output, '*', '*', 'crop*.png'))
        print(f"Deleting {len(old_crops)} old crops...")
        for old_crop in tqdm(old_crops):
            os.remove(old_crop)

        cameras = IGSceneDataset({'data': {'split': args.dataset}}).split
        args_dict = args.__dict__.copy()
        args_list = []
        for camera in cameras:
            args_dict['camera'] = camera
            args_list.append(argparse.Namespace(**args_dict))

        print("Cropping objects...")
        if args.processes == 0:
            r = []
            for a in tqdm(args_list):
                r.append(crop_images(a))
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(crop_images, args_list), total=len(args_list)))

    # render and preprocess obj
    if not args.skip_render or not args.skip_watertight or not args.skip_ldif or not args.skip_mgn:
        args_dict = args.__dict__.copy()
        args_dict['spacing'] = args.bbox / 32
        args_dict['bbox'] = ' '.join([str(-args.bbox / 2), ] * 3 + [str(args.bbox / 2), ] * 3)
        print(f"bbox: [{args_dict['bbox']}] spacing: {args_dict['spacing']}")

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

        print("Rendering and making mesh watertight...")
        if args.processes == 0:
            r = []
            for a in tqdm(args_list):
                r.append(preprocess_obj(a))
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(preprocess_obj, args_list), total=len(args_list)))


if __name__ == "__main__":
    main()
