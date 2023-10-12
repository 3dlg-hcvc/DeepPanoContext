import os
import json
import argparse
import pickle
import numpy as np
from PIL import Image
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import shutil
from shapely.geometry import Polygon, Point, MultiPoint
import shapely
from glob import glob
import traceback

from configs.data_config import NYU40CLASSES, COMMON25CLASSES, NYU40_2_COMMON25
from data.s3d_metadata.labels import IDX_TO_LABLE, LABEL_TO_IDX
from utils.relation_utils import RelationOptimization
from utils.render_utils import seg2obj, is_obj_valid
from .igibson_utils import hash_split, IGScene
from .layout_utils import horizon_layout_gt_from_scene_data, \
        manhattan_world_layout_from_pix_layout
from .transform_utils import bdb3d_corners, IGTransform
# from utils.basic_utils import write_json, read_pkl, write_pkl


data_dir = "/datasets/external/Structured3D/data/{scene_name}/2D_rendering/{room_id}/panorama"

def _render_scene_fail_remove(args):
    output_folder = os.path.join(args.output, args.scene_name, args.room_id)
    try:
        data_path = _render_scene(args)
    except Exception as err:
        data_path = None
        traceback.print_exc()
        if args.strict:
            raise err
    if not data_path:
        tqdm.write(f"Failed to generate {args.scene_name}/{args.room_id}")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    else:
        return data_path


def _render_scene(args):
    # preparation
    scene_name = args.scene_name
    room_id = args.room_id
    
    output_folder = os.path.join(args.output, scene_name, room_id)
    os.makedirs(output_folder, exist_ok=True)
    
    # resize images
    if not os.path.exists(os.path.join(args.output, scene_name, room_id, "rgb.png")):
        rgb_path = os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "full/rgb_rawlight.png")
        Image.open(rgb_path).convert("RGB").resize((1024, 512)).save(os.path.join(args.output, scene_name, room_id, "rgb.png"))
        
        inst_path = os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "full/instance.png")
        Image.open(inst_path).resize((1024, 512), Image.NEAREST).save(os.path.join(args.output, scene_name, room_id, "seg.png"))
        
        sem_path = os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "full/semantic.png")
        Image.open(sem_path).resize((1024, 512), Image.NEAREST).save(os.path.join(args.output, scene_name, room_id, "sem.png"))
    
    layout_file = os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "layout.txt")
    camera_file = os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "camera_xyz.txt")
    bbox_file = f"/datasets/external/Structured3D/data/{scene_name}/bbox_3d.json"
    manhattan_pix = np.loadtxt(layout_file, dtype=np.int32)
    bdb3ds = json.load(open(bbox_file))
    inst_id2idx = dict()
    for idx, bdb3d in enumerate(bdb3ds):
        inst_id2idx[bdb3d.get('ID')] = idx
    
    camera_xyz = np.loadtxt(camera_file, dtype=np.float) / 1000.
    cam3d2world = np.array(
        [[1, 0, 0, camera_xyz[0]],
        [0, 0, 1, camera_xyz[1]],
        [0, -1, 0, camera_xyz[2]],
        [0, 0, 0, 1]]
    )
    world2cam3d = np.linalg.inv(cam3d2world)
    cam_pos = cam3d2world[:3, 3]
    cam_height = cam3d2world[2, 3]
    cam_view = np.array([0, 1, 0])
    camera = {
            "id": f"{scene_name}/{room_id}",
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
            'name': room_id,
            'scene': scene_name,
            'camera': camera,
            'image_path': {
                'rgb': os.path.join(args.output, scene_name, room_id, "rgb.png"),
                'seg': os.path.join(args.output, scene_name, room_id, "seg.png"),
                'sem': os.path.join(args.output, scene_name, room_id, "sem.png")
            }
        }
    skip_info = f"Skipped camera {data['name']} of {data['scene']}: "
    
    # generate camera layout and check if the camaera is valid
    layout = {'manhattan_pix': manhattan_pix}
    data['layout'] = layout
    if args.world_lo:
        layout['manhattan_world'] = manhattan_world_layout_from_pix_layout(IGScene(data), cam_height)
    if args.horizon_lo:
        layout['horizon'] = horizon_layout_gt_from_scene_data(data)
    
    # get object params
    semantic = np.array(Image.open(os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "full/semantic.png")))
    instance = cv2.imread(os.path.join(data_dir.format(scene_name=scene_name, room_id=room_id), "full/instance.png"), cv2.IMREAD_UNCHANGED)
    instance_ids = np.unique(instance)[:-1] # remove 65535 for background
    
    objs = {}
    for inst_id in instance_ids:
        bdb3d = bdb3ds[inst_id2idx[inst_id]]
        labels, occurences = np.unique(semantic[np.where(instance==inst_id)], return_counts=True)
        label = labels[max(enumerate(occurences), key=lambda x: x[1])[0]]
        cat = IDX_TO_LABLE[label] # NYU40
        if args.cls_mode == 'cls25':
            if cat in NYU40_2_COMMON25:
                cat = NYU40_2_COMMON25[cat]
            if cat not in COMMON25CLASSES:
                continue
        obj_dict = {
            "id": inst_id,
            "classname": cat,
            "label": OBJCLASSES.index(cat),
            "is_fixed": True,
        }
        obj_dict["bdb3d"] = {
            "centroid": np.array(bdb3d["centroid"], dtype=np.float32) / 1000.,
            "basis": np.array(bdb3d["basis"], dtype=np.float32).T,
            "size": np.array(bdb3d["coeffs"], dtype=np.float32) * 2 / 1000., # scale for unit cube
        }
        objs[inst_id] = obj_dict

    # # get object layout
    # object_layout = []
    # for obj in objs.values():
    #     corners = bdb3d_corners(obj['bdb3d'])
    #     corners2d = corners[(0, 1, 3, 2), :2]
    #     obj2d = Polygon(corners2d)
    #     object_layout.append(obj2d)
    # object_layout = shapely.ops.cascaded_union(object_layout)
    # plot_layout(object_layout)

    # # extract object params
    data['objs'] = []
    for inst_id in objs:
        # if obj_id not in objs.keys():
        #     continue
        obj_dict = objs[inst_id].copy()

        # get object bdb2d
        seg_obj_info = seg2obj(instance, inst_id)
        obj_dict.update(seg_obj_info)
        if not is_obj_valid(obj_dict):
            continue

        # rotate camera to recenter bdb3d
        recentered_trans = IGTransform.level_look_at(data, obj_dict['bdb3d']['centroid'])
        corners = recentered_trans.world2campix(bdb3d_corners(obj_dict['bdb3d']))
        full_convex = MultiPoint(corners).convex_hull

        # filter out objects by ratio of visible part
        contour = obj_dict['contour']
        contour_points = np.stack([contour['x'], contour['y']]).T
        visible_convex = MultiPoint(contour_points).convex_hull
        if visible_convex.area / full_convex.area < 0.2:
            continue

        data['objs'].append(obj_dict)

    if not data['objs']:
        print(f"{skip_info}no object in the frame")
        return None

    # construction IGScene
    # s3d_scene = IGScene(data)

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
    parser = argparse.ArgumentParser(description='Prepare Structure3D')
    parser.add_argument('--scene', dest='scene_name',
                        type=str, default=None,
                        help='The name of the scene to load')
    parser.add_argument('--output', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/s3d',
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
    parser.add_argument('--cls_mode', type=str, default='cls40',
                        help='Types of object classes: cls40/cls25')
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
    global OBJCLASSES
    OBJCLASSES = NYU40CLASSES
    if args.cls_mode == 'cls25':
        args.output = f"{args.output}_{args.cls_mode}"
        OBJCLASSES = COMMON25CLASSES

    assert args.vertical_fov is not None or args.cam_pitch != 0, \
        "cam_pitch not supported for panorama rendering"
    assert all(r in ['rgb', 'normal', 'seg', 'sem', 'depth', '3d'] for r in args.render_type), \
        "please check render type setting"
    assert args.vertical_fov is not None or not any(r in args.render_type for r in ['normal']), \
        "render type 'normal' not supported for panorama"
    
    scenes = [l.strip() for l in open("/local-scratch/qiruiw/research/DeepPanoContext/data/s3d_metadata/scenes.txt")][1:]

    # begin rendering
    data_paths = None
    if not args.split:
        args_list = []
        args_dict = args.__dict__.copy()
        for scene in scenes:
            scene_name, room_id = scene.split(",")
            args_dict['scene_name'] = scene_name
            args_dict['room_id'] = room_id
            args_list.append(argparse.Namespace(**args_dict))
        print(f"{len(args_list)} scenes to be rendered")

        if args.processes == 0:
            data_paths = []
            for a in tqdm(args_list):
                data_paths.append(_render_scene_fail_remove(a))
        else:
            with Pool(processes=args.processes) as p:
                data_paths = list(tqdm(p.imap(_render_scene_fail_remove, args_list), total=len(args_list)))

    if not args.skip_split:
        if data_paths is None:
            data_paths = sorted(glob(os.path.join(args.output, '*', '*', 'data.pkl')))
        # split dataset
        split = {'train': [], 'test': []}
        scenes = {'train': set(), 'test': set()}
        for camera in data_paths:
            if camera is None: continue
            scene_name, room_id = camera.split('/')[-3:-1]
            scene = f"{scene_name}/{room_id}"
            is_train = hash_split(args.train, scene)
            path = os.path.join(*camera.split('/')[-3:])
            if is_train:
                split['train'].append(path)
                scenes['train'].add(scene)
            else:
                split['test'].append(path)
                scenes['test'].add(scene)

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
