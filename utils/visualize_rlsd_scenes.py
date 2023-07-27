import os
import json
import argparse
import pickle
# from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil

from models.detector.dataset import register_igibson_detection_dataset
from .igibson_utils import IGScene
from .image_utils import save_image, show_image
# from models.pano3d.dataloader import IGSceneDataset
from .visualize_utils import IGVisualizer


def visualize_camera(args):
    arch_id, pano_id = args.scene_name.split("/")
    scene_folder = os.path.join(args.dataset, args.scene_name) if args.scene_name is not None else args.dataset
    camera_folder = os.path.join(scene_folder, args.task_id)
    if not os.path.exists(os.path.join(camera_folder, 'data.pkl')): return
    scene = IGScene.from_pickle(camera_folder)
    
    if not ('objs' not in scene.data or not scene['objs'] or 'bdb3d' not in scene['objs'][0]):
        rotx90 = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        for obj in scene['objs']:
            bdb3d = obj['bdb3d']
            bdb3d['basis'] = bdb3d['basis'] @ rotx90
    
    visualizer = IGVisualizer(scene, gpu_id=args.gpu_id, debug=args.debug)

    # if not args.skip_render:
    #     render = visualizer.render(background=200)

    image = visualizer.image('rgb')
    image = visualizer.layout(image, total3d=False)
    image = visualizer.objs3d(image, bbox3d=True, axes=False, centroid=False, info=False, thickness=1)
    if not args.show:
        save_path = os.path.join(scene_folder, args.task_id, 'det3d.png')
        save_image(image, save_path)
        # save_image(image, './det3d.png')
        save_dir = f"/project/3dlg-hcvc/rlsd/www/annotations/docs/viz_v2/{args.task_id}/w_pano_camera_n_anno"
        os.makedirs(save_dir, exist_ok=True)
        save_image(image, os.path.join(save_dir, f"{pano_id}.det3d.png"))
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
        #     save_path = os.path.join(scene_folder, args.task_id, 'render.png')
        #     save_image(render, save_path)
        save_path = os.path.join(scene_folder, args.task_id, 'visual.png')
        save_image(image, save_path)
        # save_image(image, './visual.png')
        save_dir = f"/project/3dlg-hcvc/rlsd/www/annotations/docs/viz_v2/{args.task_id}/w_pano_camera_n_anno"
        os.makedirs(save_dir, exist_ok=True)
        save_image(image, os.path.join(save_dir, f"{pano_id}.visual.png"))
    
    save_dir = f"/project/3dlg-hcvc/rlsd/www/annotations/docs/viz_v2/{args.task_id}/rooms_layout2d"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(f"/project/3dlg-hcvc/rlsd/data/psu/rlsd/{arch_id}/{pano_id}/rooms.png", save_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson scenes.')
    parser.add_argument('--dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/rlsd',
                        help='The path of the rlsd dataset')
    # parser.add_argument('--igibson_obj_dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/igibson_obj',
    #                     help='The path of the iGibson object dataset')
    parser.add_argument('--full_pano_id', type=str, default=None,
                        help='The name of the scene to visualize')
    parser.add_argument('--task_id', type=str, default=None,
                        help='The task_id of the camera to visualize')
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
    register_igibson_detection_dataset(args.dataset)
    

    if args.full_pano_id is not None and args.task_id is not None:
        args_dict = args.__dict__.copy()
        house_id, level_id, pano_id = args.full_pano_id.split('_')
        args_dict['scene_name'] = f'{house_id}_{level_id}/{pano_id}'
        visualize_camera(argparse.Namespace(**args_dict))
    elif args.full_pano_id is None and args.task_id is None:
        task_pano_mapping = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/task_pano_mapping.json"))
        task_ids = list(task_pano_mapping.keys())
        args_dict = args.__dict__.copy()
        args_list = []
        for task_id in task_ids:
            args_dict['task_id'] = task_id
            full_pano_id = task_pano_mapping[task_id]
            house_id, level_id, pano_id = full_pano_id.split('_')
            args_dict['scene_name'] = f'{house_id}_{level_id}/{pano_id}'
            if not os.path.exists(os.path.join(args.dataset, args_dict['scene_name'], task_id, 'data.pkl')):
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

