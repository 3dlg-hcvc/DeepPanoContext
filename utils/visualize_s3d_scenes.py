import os
import json
import argparse
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil
from glob import glob

from models.detector.dataset import register_detection_dataset
from .igibson_utils import IGScene
from .image_utils import save_image, show_image
# from models.pano3d.dataloader import IGSceneDataset
from .visualize_utils import IGVisualizer


def visualize_camera(args):
    # arch_id, pano_id = args.scene.split("/")
    scene_folder = os.path.join(args.dataset, args.scene) if args.scene is not None else args.dataset
    if not os.path.exists(os.path.join(scene_folder, 'data.pkl')): return
    scene = IGScene.from_pickle(scene_folder)
    
    visualizer = IGVisualizer(scene, gpu_id=args.gpu_id, debug=args.debug)

    # if not args.skip_render:
    #     render = visualizer.render(background=200)

    image = visualizer.image('rgb')
    image = visualizer.layout(image, total3d=False)
    image = visualizer.objs3d(image, bbox3d=True, axes=True, centroid=False, info=False, thickness=1)
    if not args.show:
        save_path = os.path.join(scene_folder, 'det3d.png')
        save_image(image, save_path)
        # save_image(image, './det3d.png')
        # save_dir = f"/project/3dlg-hcvc/rlsd/www/annotations/docs/viz_v2/w_pano_camera_n_anno"
        # os.makedirs(save_dir, exist_ok=True)
        # save_image(image, os.path.join(save_dir, f"{pano_id}.png"))
    # image = visualizer.bfov(image, thickness=1, include=('walls', 'objs'))
    image = visualizer.bdb2d(image)

    if args.show:
        # if not args.skip_render:
        #     show_image(render)
        #     if 'K' in scene['camera']:
        #         birds_eye = visualizer.render(background=200, camera='birds_eye')
        #         show_image(birds_eye)
        #         up_down = visualizer.render(background=200, camera='up_down')
        #         show_image(up_down)
        show_image(image)
    else:
        # if not args.skip_render:
        #     save_path = os.path.join(scene_folder, 'render.png')
        #     save_image(render, save_path)
        save_path = os.path.join(scene_folder, 'visual.png')
        save_image(image, save_path)
        # save_image(image, './visual.png')
        # save_dir = f"/project/3dlg-hcvc/rlsd/www/annotations/docs/viz_v2/w_pano_camera_n_anno/"
        # os.makedirs(save_dir, exist_ok=True)
        # save_image(image, os.path.join(save_dir, f"{pano_id}.png"))


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson scenes.')
    parser.add_argument('--dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/s3d',
                        help='The path of the rlsd dataset')
    parser.add_argument('--scene_name', type=str, default=None,
                        help='The name of the scene to visualize')
    parser.add_argument('--room_id', type=str, default=None,
                        help='The room_id of the camera to visualize')
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
    

    if args.scene_name is not None and args.room_id is not None:
        args_dict = args.__dict__.copy()
        args_dict['scene'] = f'{args.scene_name}/{args.room_id}'
        visualize_camera(argparse.Namespace(**args_dict))
    elif args.scene_name is None and args.room_id is None:
        args_dict = args.__dict__.copy()
        args_list = []
        scenes = sorted(glob(os.path.join(args.dataset, '*', '*')))
        for scene in scenes:
            scene_name, room_id = scene.split('/')[-2:]
            args_dict['scene'] = f'{scene_name}/{room_id}'
            if not os.path.exists(os.path.join(args.dataset, args_dict['scene'], 'data.pkl')):
                continue
            args_list.append(argparse.Namespace(**args_dict))
        
        if args.processes == 0:
            for a in tqdm(args_list):
                visualize_camera(a)
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(visualize_camera, args_list), total=len(args_list)))
    else:
        raise Exception('Should specify both scene and task_id for visualizing single camera. ')

if __name__ == "__main__":
    main()

