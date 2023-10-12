import os
import json
import jsonlines
import argparse
import math
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import shutil
import shapely
from shapely.geometry import Polygon, Point, MultiPoint
from glob import glob
import traceback

from configs.data_config import PSU45CLASSES, IG56CLASSES, CUSTOM2RLSD, RLSD32_2_IG56, COMMON25CLASSES, RLSD32CLASSES
from utils.relation_utils import RelationOptimization
from utils.render_utils import seg2obj, is_obj_valid
from .rlsd_utils import create_data_splits, encode_rgba, prepare_images
from .igibson_utils import IGScene
from .layout_utils import scene_layout_from_rlsd_arch, room_layout_from_rlsd_scene, \
    manhattan_pix_layout_from_rlsd_room, \
    manhattan_world_layout_from_room_layout, horizon_layout_gt_from_scene_data
from .transform_utils import bdb3d_corners, IGTransform


issues = {key:[] for key in ["duplicate_points", "duplicate_x", "over_large_objects", "zero_obj_dim", "outside_house", "mask_missing", "close_to_wall"]}
issues["close_to_wall"] = {key:[] for key in ["0.5", "0.3", "0.1"]}
model_paths = set()
missing_3dw = set()
model2cat = {}
OBJCLASSES = None


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


def _render_scene(args):
    # preparation
    scene_name, scene_source = args.scene_name, args.scene_source
    house_id = scene_name.split("_")[0]
    pano_id = scene_name.split("/")[-1]
    task_id = args.task_id
    full_task_id = f'{scene_name}/{task_id}'
    task_file = f"/project/3dlg-hcvc/rlsd/data/annotations/complete_task_json/{task_id}.json"
    task_json = json.load(open(task_file))
    panos = json.load(open("/project/3dlg-hcvc/rlsd/data/mp3d/pano_objects_mapping.json"))
    
    output_folder = os.path.join(args.output, scene_name, task_id)
    os.makedirs(output_folder, exist_ok=True)
    # resize images
    prepare_images(args)

    # generate scene layout
    rooms, rooms_scale = scene_layout_from_rlsd_arch(args)
    if not rooms:
        raise Exception('Layout not valid!')
    
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
            }
        }
    skip_info = f"Skipped camera {data['name']} of {data['scene']}: "
    plot_path = os.path.join(args.output, scene_name)
    room, wall_ind_map, distance_wall = room_layout_from_rlsd_scene(camera, rooms, panos, plot_path)
    if room is None:
        issues["outside_house"].append(full_task_id)
        print(skip_info + "room layout generation failed")
        return
    if distance_wall < 0.5:
        issues["close_to_wall"]["0.5"].append(f"{full_task_id}/{distance_wall}")
        # print(f"{full_task_id} close to wall ({distance_wall:.3f} < 0.5)")
    if distance_wall < 0.3:
        issues["close_to_wall"]["0.3"].append(f"{full_task_id}/{distance_wall}")
    if distance_wall < 0.1:
        issues["close_to_wall"]["0.1"].append(f"{full_task_id}/{distance_wall}")
        print(skip_info + "room layout generation failed")
        return
    data['room'] = room
    
    # generate camera layout and check if the camaera is valid
    layout = {'manhattan_pix': manhattan_pix_layout_from_rlsd_room(camera, room, args.room_mode, full_task_id, issues)}
    data['layout'] = layout
    if layout['manhattan_pix'] is None:
        print(skip_info + "manhattan pixel layout generation failed")
        return
    if args.world_lo:
        layout['manhattan_world'] = manhattan_world_layout_from_room_layout(room)
    if args.horizon_lo:
        layout['horizon'] = horizon_layout_gt_from_scene_data(data)
    
    # get object params
    scene_json = task_json["sceneJson"]
    objects = scene_json["scene"]["object"]
    objects = {obj["id"]: obj for obj in objects}
    mask_infos = {mask_info["id"]: mask_info for mask_info in scene_json["maskInfos"] if mask_info}
    mask_assignments = scene_json["maskObjectAssignments"]
    obj2masks = {}
    for assign in mask_assignments:
        obj_id = assign["objectInstanceId"]
        obj_masks = obj2masks.setdefault(obj_id, [])
        obj_masks.append(assign["maskId"])
    objs = {}
    rotx90 = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    for obj_id in objects:
        if obj_id not in obj2masks:
            continue
        obj = objects[obj_id]
        if np.any((np.array(obj["obb"]["axesLengths"])-np.array(rooms_scale)) > 1):
            if full_task_id not in issues["over_large_objects"]:
                issues["over_large_objects"].append(full_task_id)
            continue
        if np.any(np.array(obj["obb"]["axesLengths"]) < 1e-6):
            issues["zero_obj_dim"].append(f"{full_task_id}/{obj_id}/{obj['modelId']}")
            continue
        mask_ids = obj2masks[obj_id]
        # categories = [mask_infos[mask_id]["label"] for mask_id in mask_ids if mask_id in mask_infos]
        categories = []
        for mask_id in mask_ids:
            if mask_id not in mask_infos:
                issues["mask_missing"].append(f"{full_task_id}/{mask_id}")
                continue
            cat = mask_infos[mask_id]["label"].lower()
            if cat not in RLSD32CLASSES: # and mask_infos[mask_id]["type"] != "mask":
                try:
                    cat = CUSTOM2RLSD[cat]
                except:
                    continue
            if args.cls_mode == 'cls25':
                if cat not in COMMON25CLASSES: 
                    continue
            if args.model_mode == 'ig':
                cat = RLSD32_2_IG56[cat]
            categories.append(cat)
        if not categories:
            continue
        model_source, model_name = obj["modelId"].split('.')
        model_path = f'{categories[0]}/{model_name}'
        if model_source == 'wayfair':
            model_path = f'/datasets/internal/models3d/wayfair/wayfair_models_cleaned/{model_name}/{model_name}.glb'
        elif model_source == '3dw':
            model_path = f'/project/3dlg-hcvc/rlsd/data/3dw/objmeshes/{model_name}/{model_name}.obj'
            if not os.path.exists(model_path):
                missing_3dw.add(model_name)
        else:
            raise NotImplementedError
        model_paths.add(model_path)
        obj_dict = {
            "id": obj_id,
            "mask_ids": mask_ids,
            "index": obj["index"],
            "classname": categories,
            "label": [OBJCLASSES.index(cat) for cat in categories],
            "model_name": obj["modelId"],
            "model_path": model_path,
            "is_fixed": True,
        }
        obj_dict["bdb3d"] = {
            "centroid": np.array(obj["obb"]["centroid"], dtype=np.float32),
            "basis": np.array(obj["obb"]["normalizedAxes"], dtype=np.float32).reshape(3, 3).T @ rotx90,
            "size": np.array(obj["obb"]["axesLengths"], dtype=np.float32),
        }
        obj_dict['bdb3d']['size'][[1, 2]] = obj_dict['bdb3d']['size'][[2, 1]]
        objs[obj_id] = obj_dict

    # # get object layout
    # object_layout = []
    # for obj in objs.values():
    #     corners = bdb3d_corners(obj['bdb3d'])
    #     corners2d = corners[(0, 1, 3, 2), :2]
    #     obj2d = Polygon(corners2d)
    #     object_layout.append(obj2d)
    # object_layout = shapely.ops.cascaded_union(object_layout)
    # # plot_layout(object_layout)

    # extract object params
    data['objs'] = []
    inst_path = os.path.join(args.output, scene_name, "seg.png")
    inst_seg = encode_rgba(np.array(Image.open(inst_path)))
    if args.img_mode == 'syn':
        obj2insts = {}
        label_csv_path = f"/project/3dlg-hcvc/rlsd/data/annotations/equirectangular_instance/{task_id}/{task_id}.scene.objectId.csv"
        label_df = pd.read_csv(label_csv_path)
        for obj_id in obj2masks:
            obj2insts[obj_id] = label_df[label_df.label == obj_id].index.tolist()
    else:
        obj2insts = obj2masks
    for obj_id in objs:
        obj_dict = objs[obj_id].copy()
        valid_mask_ids = [mask_id for mask_id in obj_dict["mask_ids"] if mask_id in mask_infos and "type" in mask_infos[mask_id] and mask_infos[mask_id]["type"] == "mask"]
        if not valid_mask_ids:
            continue

        # get object bdb2d
        encode_inst_ind = [inst_idx*256+255 for inst_idx in obj2insts[obj_id]]
        seg_obj_info = seg2obj(inst_seg, encode_inst_ind)
        if not seg_obj_info:
            continue
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
        
    for obj in data['objs']:
        obj_id = obj['id']
        obj.update({'obj_parent': -1, 'floor_supp': 0, 'ceil_supp': 0, 'wall_supp': 0, 'wall_parent': -1})
        if 'parentId' not in objects[obj_id]:
            continue
        parent_id = objects[obj_id]['parentId']
        parent_idx = list(filter(lambda i: data['objs'][i]['id'] == parent_id, range(len(data['objs']))))
        if parent_idx:
            obj['obj_parent'] = parent_idx[0]
        else:
            if parent_id[-1] == 'f':
                obj['floor_supp'] = 1
            elif parent_id[-1] == 'c':
                obj['ceil_supp'] = 1
            elif len(parent_id.split('_')) == 3: # wall
                obj['wall_supp'] = 1
                if int(parent_id.split('_')[1]) == data['room_idx']:
                    obj['wall_parent'] = wall_ind_map.get(int(parent_id.split('_')[-1]), -1)

    if not data['objs']:
        print(f"{skip_info}no object in the frame")
        # return None

    # construction IGScene
    # rlsd_scene = IGScene(data)

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
    parser.add_argument('--output', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/rlsd',
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
    parser.add_argument('--train', type=float, default=0.8,
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
    parser.add_argument('--split_by', default='pano',
                        help='Specify split strategy: house/region/pano')
    parser.add_argument('--room_mode', type=str, default='regions',
                        help='Types of room layout')
    parser.add_argument('--img_mode', type=str, default='real',
                        help='Types of images: real/syn/mix')
    parser.add_argument('--cls_mode', type=str, default='cls45',
                        help='Types of object classes: cls45/cls25')
    parser.add_argument('--model_mode', type=str, default='rlsd',
                        help='Types of images: rlsd/ig')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from existing renders')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    parser.add_argument('--crop_width', default=None, type=int,
                        help='Width of image cropped of ground truth 2d bounding box')
    parser.add_argument('--relation', default=False, action='store_true',
                        help='Generate relationships')
    parser.add_argument('--skip_split', default=False, action='store_true',
                        help='Skip train/test split')
    args = parser.parse_args()
    args.output = f"{args.output}_{args.img_mode}"
    global OBJCLASSES
    OBJCLASSES = PSU45CLASSES
    if args.cls_mode == 'cls25':
        args.output = f"{args.output}_{args.cls_mode}"
        OBJCLASSES = COMMON25CLASSES
    if args.model_mode == 'ig':
        args.output = f"{args.output}_{args.model_mode}"
        OBJCLASSES = IG56CLASSES

    assert args.vertical_fov is not None or args.cam_pitch != 0, \
        "cam_pitch not supported for panorama rendering"
    assert all(r in ['rgb', 'normal', 'seg', 'sem', 'depth', '3d'] for r in args.render_type), \
        "please check render type setting"
    assert args.vertical_fov is not None or not any(r in args.render_type for r in ['normal']), \
        "render type 'normal' not supported for panorama"
        
    task_pano_mapping = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/task_pano_mapping.json"))
    
    invalid_rt_anno = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/invalid_room_type_annotation.json"))
    invalid_tasks = invalid_rt_anno["stairs"] + invalid_rt_anno["outdoor"]
    for full_task_id in invalid_tasks:
        del task_pano_mapping[full_task_id.split('/')[1]]
    
    task_ids = list(task_pano_mapping.keys())

    # begin rendering
    data_paths = None
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

    if args.img_mode == 'real' and \
        args.model_mode == 'rlsd' and \
        args.cls_mode == 'cls45' and \
        not args.split:
        with open("/project/3dlg-hcvc/rlsd/data/annotations/annotation_issues.json", 'w') as f:
            json.dump(issues, f, indent=4)
        with open("/project/3dlg-hcvc/rlsd/data/annotations/unique_shapes.txt", 'w') as f:
            for p in model_paths:
                f.write(f"{p}\n")
        with open("/project/3dlg-hcvc/rlsd/data/annotations/missing_3dw.txt", 'w') as f:
            for m in missing_3dw:
                f.write(f"{m}\n")

    if not args.skip_split:
        if data_paths is None:
            data_paths = sorted(glob(os.path.join(args.output, '*', '*', '*', 'data.pkl')))
        # split dataset
        split, scenes = create_data_splits(args, data_paths)

        print(f"{len(scenes['train']) + len(scenes['test'])} scenes, "
            f"{len(scenes['train'])} train scenes, "
            f"{len(scenes['test'])} test scenes, "
            f"{len(split['train'])} train cameras, "
            f"{len(split['test'])} test cameras")

        for k, v in split.items():
            v.sort()
            with open(os.path.join(args.output, k + '.json'), 'w') as f:
                json.dump(v, f, indent=4)


if __name__ == "__main__":
    main()
