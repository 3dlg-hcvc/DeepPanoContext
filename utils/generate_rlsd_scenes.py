import os
import json
import jsonlines
import argparse
import math
import pickle
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import shutil
from shapely.geometry import Polygon, Point, MultiPoint
import shapely
from glob import glob
import traceback

from configs.data_config import IG56CLASSES
from utils.relation_utils import RelationOptimization
from utils.render_utils import seg2obj, is_obj_valid
from .igibson_utils import hash_split, IGScene
from .layout_utils import scene_layout_from_rlsd_arch, room_layout_from_rlsd_scene, \
    manhattan_pix_layout_from_rlsd_room, \
    manhattan_world_layout_from_room_layout, horizon_layout_gt_from_scene_data
from .transform_utils import bdb3d_corners, IGTransform
# from utils.basic_utils import write_json, read_pkl, write_pkl


rgb_dir = "/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_rgb_panos"
inst_dir = "/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_instance_panos"


def _render_scene_fail_remove(args):
    output_folder = os.path.join(args.output, args.scene_name, args.task_id)
    try:
        data_path = _render_scene(args)
    except Exception as err:
        data_path = None
        traceback.print_exc()
        if args.strict:
            raise err
    if not data_path:
        tqdm.write(f"Failed to generate {args.scene_name}")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    else:
        return data_path


def encode_rgba(arr):
    arr = arr[:,:,0]*(pow(2,24)) + arr[:,:,1]*(pow(2,16)) + arr[:,:,2]*(pow(2,8)) + arr[:,:,3]
    return arr


def _render_scene(args):
    # preparation
    scene_name, scene_source = args.scene_name, args.scene_source
    house_id = scene_name.split("_")[0]
    pano_id = scene_name.split("/")[-1]
    task_id = args.task_id
    task_file = f"/project/3dlg-hcvc/rlsd/data/annotations/task_json/{task_id}.json"
    task_json = json.load(open(task_file))
    panos = json.load(open("/project/3dlg-hcvc/rlsd/data/mp3d/pano_objects_mapping.json"))
    
    output_folder = os.path.join(args.output, scene_name, task_id)
    os.makedirs(output_folder, exist_ok=True)
    
    # resize images
    if not os.path.exists(os.path.join(args.output, scene_name, "rgb.png")):
        rgb_path = f"{rgb_dir}/{house_id}/{pano_id}.png"
        Image.open(rgb_path).convert("RGB").resize((1024, 512)).save(os.path.join(args.output, scene_name, "rgb.png"))
        
        inst_path = f"{inst_dir}/{house_id}/{pano_id}.objectId.encoded.png"
        Image.open(inst_path).resize((1024, 512), Image.NEAREST).save(os.path.join(args.output, scene_name, "seg.png"))
        
        depth_path = f"{inst_dir}/{house_id}/{pano_id}.depth.png"
        Image.open(depth_path).resize((1024, 512), Image.NEAREST).save(os.path.join(args.output, scene_name, "depth.png"))

    # generate scene layout
    rooms = scene_layout_from_rlsd_arch(args)
    if not rooms:
        raise Exception('Layout not valid!')
    
    # get object params
    scene_json = task_json["sceneJson"]
    objects = scene_json["scene"]["object"]
    objects = {obj["id"]: obj for obj in objects}
    mask_infos = {mask_info["id"]: mask_info for mask_info in scene_json["maskInfos"]}
    mask_assignments = scene_json["maskObjectAssignments"]
    # obj2masks = {assign["objectInstanceId"]: assign["maskId"] for assign in mask_assignments}
    obj2masks = {}
    for assign in mask_assignments:
        obj_id = assign["objectInstanceId"]
        if obj_id not in obj2masks:
            obj2masks[obj_id] = [assign["maskId"]]
        else:
            obj2masks[obj_id].append(assign["maskId"])
    objs = {}
    for obj_id in objects:
        if obj_id not in obj2masks:
            continue
        mask_ids = obj2masks[obj_id]
        categories = [mask_infos[mask_id]["label"] for mask_id in mask_ids if mask_id in mask_infos]
        # for cat in categories:
        #     if cat not in RLSD32CLASSES:
        #         import pdb; pdb.set_trace()
        obj = objects[obj_id]
        model_source, model_name = obj["modelId"].split('.')
        if model_source == 'wayfair':
            model_path = f'/datasets/external/3dfront/3D-FUTURE-model/{model_name}/raw_model.obj'
        elif model_source == '3dw':
            model_path = f'/project/3dlg-hcvc/rlsd/data/3dw/objmeshes_local/{model_name}/{model_name}.obj'
        else:
            raise NotImplementedError
        obj_dict = {
            "id": obj_id,
            "mask_ids": mask_ids,
            "index": obj["index"],
            "classname": categories,
            # "label": [RLSD48CLASSES.index(cat) for cat in categories],
            "model_name": obj["modelId"],
            "model_path": model_path,
            "is_fixed": True,
        }
        obj_dict["bdb3d"] = {
            "centroid": np.array(obj["obb"]["centroid"], dtype=np.float32),
            "basis": np.array(obj["obb"]["normalizedAxes"], dtype=np.float32).reshape(3, 3).T,
            "size": np.array(obj["obb"]["axesLengths"], dtype=np.float32),
        }
        objs[obj_id] = obj_dict

    # get object layout
    object_layout = []
    for obj in objs.values():
        corners = bdb3d_corners(obj['bdb3d'])
        corners2d = corners[(0, 1, 3, 2), :2]
        obj2d = Polygon(corners2d)
        object_layout.append(obj2d)
    object_layout = shapely.ops.cascaded_union(object_layout)
    # plot_layout(object_layout)
    
    with jsonlines.open(f"/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_camera_poses/{house_id}.jsonl") as cameras:
        for c in cameras:
            if c["id"] == pano_id:
                camera = c
                break
    
    recover = np.linalg.inv(np.array([[math.cos(np.pi/2), 0, math.sin(np.pi/2)],
           [0, 1, 0],
           [-math.sin(np.pi/2), 0, math.cos(np.pi/2)]]))
            
    cam3d2world = np.array(camera["camera"]["extrinsics"]).reshape(4, 4)
    cam3d2world[:3, :3] = cam3d2world[:3, :3] @ recover
    world2cam3d = np.linalg.inv(cam3d2world)
    cam_pos = cam3d2world[:3, 3]
    cam_height = cam3d2world[2, 3]
    cam_view = np.array([0, 1, 0])
    camera = {
            "id": pano_id,
            'height': 512,
            'width': 1024,
            "pos": cam_pos,
            "view_dir": cam_view,
            "target": cam_pos + cam_view,
            "up": np.array([0, 0, 1], dtype=np.float32),
            "world2cam3d": world2cam3d,
            "cam3d2world": cam3d2world
        }
    data = {
            'name': task_id,
            'scene': scene_name,
            'room_idx': panos[pano_id]["region_index"],
            'camera': camera,
            'image_path': {
                'rgb': os.path.join(args.output, scene_name, "rgb.png"),
                'seg': os.path.join(args.output, scene_name, "seg.png"),
                'depth': os.path.join(args.output, scene_name, "depth.png")
            }
        }
    skip_info = f"Skipped camera {data['name']} of {data['scene']}: "
    plot_path = os.path.join(args.output, scene_name, "layout2d.png")
    # plot_path = f"/project/3dlg-hcvc/rlsd/www/annotations/docs/viz_v2/rooms_layout2d/{task_id}/layout2d.png"
    room_layout = room_layout_from_rlsd_scene(camera, rooms, panos, plot_path)
    if room_layout is None:
        print(skip_info + "room layout generation failed")
        return
    data['room'] = room_layout
    
    # generate camera layout and check if the camaera is valid
    layout = {'manhattan_pix': manhattan_pix_layout_from_rlsd_room(camera, room_layout)}
    data['layout'] = layout
    if layout['manhattan_pix'] is None:
        print(skip_info + "manhattan pixel layout generation failed")
        return
    if args.world_lo:
        layout['manhattan_world'] = manhattan_world_layout_from_room_layout(room_layout)
    if args.horizon_lo:
        layout['horizon'] = horizon_layout_gt_from_scene_data(data)

    # # render
    # render_results = render_camera(s.renderer, camera, args.render_type,
    #                                perspective, obj_groups, scene.objects_by_id)

    # # extract object params
    data['objs'] = []
    inst_path = os.path.join(args.output, scene_name, "seg.png")
    inst_seg = encode_rgba(np.array(Image.open(inst_path)))
    for obj_id in objs:
        # if obj_id not in objs.keys():
        #     continue
        obj_dict = objs[obj_id].copy()
        mask_ids = obj_dict["mask_ids"]
        valid_mask_ids = [mask_id for mask_id in mask_ids if mask_id in mask_infos and mask_infos[mask_id]["type"] == "mask"]
        if not valid_mask_ids:
            continue

        # get object bdb2d
        encode_mask_ids = [mask_id*256+255 for mask_id in mask_ids]
        seg_obj_info = seg2obj(inst_seg, encode_mask_ids)
        obj_dict.update(seg_obj_info)
        if not is_obj_valid(obj_dict):
            continue

        # # rotate camera to recenter bdb3d
        # recentered_trans = IGTransform.level_look_at(data, obj_dict['bdb3d']['centroid'])
        # corners = recentered_trans.world2campix(bdb3d_corners(obj_dict['bdb3d']))
        # full_convex = MultiPoint(corners).convex_hull

        # # REMOVE since positions of instance mask and annotated object don't fully match
        # # filter out objects by ratio of visible part
        # contour = obj_dict['contour']
        # contour_points = np.stack([contour['x'], contour['y']]).T
        # visible_convex = MultiPoint(contour_points).convex_hull
        # if visible_convex.area / full_convex.area < 0.2:
        #     continue

        data['objs'].append(obj_dict)

    if not data['objs']:
        print(f"{skip_info}no object in the frame")
        return None

    # construction IGScene
    rlsd_scene = IGScene(data)

    # # generate relation
    # if args.relation:
    #     relation_optimization = RelationOptimization(expand_dis=args.expand_dis)
    #     relation_optimization.generate_relation(ig_scene)

    # save data
    pickle_file = os.path.join(output_folder, 'data.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
        
    # if args.json:
    #     ig_scene.to_json(camera_folder)
    # camera_paths.append(os.path.join(data['scene'], data['name']))

    return pickle_file


def main():
    parser = argparse.ArgumentParser(
        description='Render RGB panorama from iGibson scenes.')
    parser.add_argument('--scene', dest='scene_name',
                        type=str, default=None,
                        help='The name of the scene to load')
    parser.add_argument('--source', dest='scene_source',
                        type=str, default='IG',
                        help='The name of the source dataset, among [IG,CUBICASA,THREEDFRONT]')
    parser.add_argument('--output', type=str, default='data/rlsd',
                        help='The path of the output folder')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generating camera pose')
    parser.add_argument('--width', type=int, default=512,
                        help='Height of output image')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of output image')
    parser.add_argument('--processes', type=int, default=0,
                        help='Number of threads')
    parser.add_argument('--renders', type=int, default=10,
                        help='Number of renders per room')
    parser.add_argument('--cam_height', type=float, default=[1.6], nargs='+',
                        help='Height of camera in meters (provide two numbers to specify range)')
    parser.add_argument('--cam_pitch', type=float, default=[0.], nargs='+',
                        help='Pitch of camera in degrees (provide two numbers to specify range)')
    parser.add_argument('--random_yaw', default=False, action='store_true',
                        help='Randomize camera yaw')
    parser.add_argument('--vertical_fov', type=float, default=None,
                        help='Fov for perspective camera in degrees')
    parser.add_argument('--render_type', type=str, default=['rgb', 'seg', 'sem', 'depth'], nargs='+',
                        help='Types of renders (rgb/normal/seg/sem/depth/3d)')
    parser.add_argument('--strict', default=False, action='store_true',
                        help='Raise exception if render fails')
    parser.add_argument('--super_sample', type=int, default=2,
                        help='Set to greater than 1 to use super_sample')
    parser.add_argument('--no_physim', default=False, action='store_true',
                        help='Do physical simulation before rendering')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Ratio of train split')
    parser.add_argument('--horizon_lo', default=False, action='store_true',
                        help='Generate Horizon format layout GT from manhattan layout')
    parser.add_argument('--json', default=False, action='store_true',
                        help='Save camera info as json too')
    parser.add_argument('--cuboid_lo', default=False, action='store_true',
                        help='Generate cuboid world frame layout from manhattan layout')
    parser.add_argument('--world_lo', default=False, action='store_true',
                        help='Generate manhatton world frame layout')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--split', default=False, action='store_true',
                        help='Split train/test dataset without rendering')
    # parser.add_argument('--random_obj', default=None, action='store_true',
    #                     help='Use the 10 objects randomization for each scene')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from existing renders')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    parser.add_argument('--crop_width', default=None, type=int,
                        help='Width of image cropped of ground truth 2d bounding box')
    parser.add_argument('--relation', default=False, action='store_true',
                        help='Generate relationships')
    args = parser.parse_args()

    assert args.vertical_fov is not None or args.cam_pitch != 0, \
        "cam_pitch not supported for panorama rendering"
    assert all(r in ['rgb', 'normal', 'seg', 'sem', 'depth', '3d'] for r in args.render_type), \
        "please check render type setting"
    assert args.vertical_fov is not None or not any(r in args.render_type for r in ['normal']), \
        "render type 'normal' not supported for panorama"

    # prepare arguments
    # scene_names = []
    # if args.scene_name is None:
    #     dataset_path = {'IG': gibson2.ig_dataset_path,
    #                     'CUBICASA': gibson2.cubicasa_dataset_path,
    #                     'THREEDFRONT': gibson2.threedfront_dataset_path}
    #     dataset_path = dataset_path[args.scene_source]
    #     dataset_path = os.path.join(dataset_path, "scenes")
    #     for n in os.listdir(dataset_path):
    #         if n != 'background' \
    #                 and os.path.isdir(os.path.join(dataset_path, n)) \
    #                 and n.endswith('_int'):
    #             scene_names.append(n)
    # else:
    #     scene_names = [args.scene_name]
        
    task_pano_mapping = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/task_pano_mapping.json"))
    task_ids = list(task_pano_mapping.keys())

    # begin rendering
    if not args.split:
        args_list = []
        args_dict = args.__dict__.copy()
        for task_id in task_ids:
            args_dict['task_id'] = task_id
            full_pano_id = task_pano_mapping[task_id]
            house_id, level_id, pano_id = full_pano_id.split('_')
            args_dict['scene_name'] = f'{house_id}_{level_id}/{pano_id}'
            args_list.append(argparse.Namespace(**args_dict))
        print(f"{len(args_list)} scenes to be rendered")

        if args.processes == 0:
            data_paths = []
            for a in tqdm(args_list):
                data_paths.append(_render_scene_fail_remove(a))
        else:
            with Pool(processes=args.processes) as p:
                data_paths = list(tqdm(p.imap(_render_scene_fail_remove, args_list), total=len(args_list)))

    # split dataset
    split = {'train': [], 'test': []}
    scenes = {'train': set(), 'test': set()}
    # cameras = glob(os.path.join(args.output, '*', '*', '*', 'data.pkl'))
    for camera in data_paths:
        if camera is None: continue
        scene_name = camera.split('/')[-3]
        is_train = hash_split(args.train, scene_name)
        path = os.path.join(*camera.split('/')[-4:])
        if is_train:
            split['train'].append(path)
            scenes['train'].add(scene_name)
        else:
            split['test'].append(path)
            scenes['test'].add(scene_name)

    print(f"{len(scenes['train']) + len(scenes['test'])} scenes, "
          f"{len(scenes['train'])} train scenes, "
          f"{len(scenes['test'])} test scenes, "
          f"{len(split['train'])} train cameras, "
          f"{len(split['test'])} test cameras")
    # 761 scenes, 551 train scenes, 210 test scenes, 594 train cameras, 227 test cameras

    for k, v in split.items():
        v.sort()
        with open(os.path.join(args.output, k + '.json'), 'w') as f:
            json.dump(v, f, indent=4)


if __name__ == "__main__":
    main()
