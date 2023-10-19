import os
import argparse
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from configs.data_config import COMMON25CLASSES, IG56_2_PSU45
from models.detector.dataset import register_detection_dataset
from utils.igibson_utils import IGScene
from utils.image_utils import save_image
from models.pano3d.dataloader import SceneDataset
from utils.visualize_utils import IGVisualizer


def update_scene(args):
    scene_folder = os.path.join(args.dataset, args.scene) if args.scene is not None else args.dataset
    camera_folder = os.path.join(scene_folder, args.id)
    scene = IGScene.from_pickle(camera_folder, args.igibson_obj_dataset)
    
    objs = scene['objs']
    objs_update = []
    for obj in objs:
        cat = obj['classname']
        if cat in IG56_2_PSU45:
            cat = IG56_2_PSU45[cat]
        if cat not in COMMON25CLASSES:
            continue
        obj['classname'] = cat
        obj['label'] = COMMON25CLASSES.index(cat)
        objs_update.append(obj)
    scene.data['objs'] = objs_update
    scene.to_pickle(camera_folder)

    visualizer = IGVisualizer(scene, gpu_id=args.gpu_id, debug=args.debug)

    image = visualizer.image('rgb')
    image = visualizer.layout(image, total3d=False)
    image = visualizer.objs3d(image, bbox3d=True, axes=False, centroid=False, info=False, thickness=1)
    save_path = os.path.join(scene_folder, args.id, 'det3d.png')
    save_image(image, save_path)
    # image = visualizer.bfov(image, include=('walls', 'objs'))
    image = visualizer.bdb2d(image)
    save_path = os.path.join(scene_folder, args.id, 'visual.png')
    save_image(image, save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson scenes.')
    parser.add_argument('--dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/igibson_cls25',
                        help='The path of the iGibson dataset')
    parser.add_argument('--igibson_obj_dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/igibson_obj',
                        help='The path of the iGibson object dataset')
    parser.add_argument('--scene', type=str, default=None,
                        help='The name of the scene to visualize')
    parser.add_argument('--id', type=str, default=None,
                        help='The id of the camera to visualize')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Save temporary files to out/tmp/ when rendering')
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip visualizing mesh GT, which is time consuming')
    parser.add_argument('--show', default=False, action='store_true',
                        help='Show visualization results instead of saving')
    args = parser.parse_args()
    register_detection_dataset(args.dataset)

    if args.scene is not None and args.id is not None:
        args_dict = args.__dict__.copy()
        update_scene(argparse.Namespace(**args_dict))
    elif args.scene is None and args.id is None:
        cameras = SceneDataset({'data': {'split': args.dataset, 'igibson_obj': args.igibson_obj_dataset}}).split
        args_dict = args.__dict__.copy()
        args_list = []
        for camera in cameras:
            camera_dirs = camera.split('/')
            if len(camera_dirs) == 4:
                args_dict['scene'], args_dict['id'] = None, camera_dirs[-2]
            else:
                args_dict['scene'], args_dict['id'] = camera_dirs[-3:-1]
            args_dict['id'] = os.path.splitext(args_dict['id'])[0]
            args_list.append(argparse.Namespace(**args_dict))
        if args.processes == 0:
            for a in tqdm(args_list):
                update_scene(a)
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(update_scene, args_list), total=len(args_list)))
    else:
        raise Exception('Should specify both scene and id for updating single camera scene. ')

if __name__ == "__main__":
    main()

