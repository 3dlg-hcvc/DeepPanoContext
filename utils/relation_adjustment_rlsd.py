import argparse
import os

import cv2
import numpy as np
from copy import deepcopy

from models.pano3d.dataloader import collate_fn
from utils.igibson_utils import IGScene
from utils.image_utils import save_image
from utils.relation_utils import RelationOptimization, visualize_relation, relation_from_bins, compare_bdb3d
from utils.transform_utils import IGTransform


def main():
    parser = argparse.ArgumentParser(
        description='Relation optimization testing.')
    parser.add_argument('--dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/rlsd_real',
                        help='The path of the RLSD dataset')
    # parser.add_argument('--igibson_obj_dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/igibson_obj',
    #                     help='The path of the iGibson object dataset')
    parser.add_argument('--full_pano_id', type=str, default=None,
                        help='The name of the scene to visualize')
    parser.add_argument('--task_id', type=str, default=None,
                        help='The task_id of the camera to visualize')
    parser.add_argument('--output', type=str, default='out/relation_adjust',
                        help='The path of the output folder')
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip visualizing mesh GT, which is time consuming')
    parser.add_argument('--show', default=False, action='store_true',
                        help='Show visualization results instead of saving to output')
    parser.add_argument('--toleration_dis', type=float, default=0.0,
                        help='Toleration distance when calculating optimization loss')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generating camera pose')
    args = parser.parse_args()
    np.random.seed(args.seed)

    house_id, level_id, pano_id = args.full_pano_id.split('_')
    scene_name = f'{house_id}_{level_id}/{pano_id}'
    scene_folder = os.path.join(args.dataset, scene_name)
    camera_folder = os.path.join(scene_folder, args.task_id)
    args.output = os.path.join(args.output, scene_name, args.task_id)
    if not os.path.exists(os.path.join(camera_folder, 'data.pkl')): return
    scene = IGScene.from_pickle(camera_folder)
    
    if isinstance(scene.data['room'], dict):
        room = scene.data['room']['id']
        del scene.data['room']
        scene.data['room'] = room
    for obj in scene['objs']:
        for k in ["mask_ids", "classname", "label"]:
            if isinstance(obj[k], list):
                obj[k] = obj[k][0] #HACK
                
    loss_weights = {
        'center': 0.0001, 'size': 1.0, 'dis': 0.01, 'ori': 0.001,
        'obj_obj_col': 0.1, 'obj_wall_col': 1.0, 'obj_floor_col': 1.0, 'obj_ceil_col': 1.0,
        'obj_obj_tch': 0.1, 'obj_wall_tch': 1.0, 'obj_floor_tch': 1.0, 'obj_ceil_tch': 1.0,
        'obj_obj_rot': 0.01, 'obj_wall_rot': 0.1,
        'obj_obj_dis': 0.01,
        'bdb3d_proj': 10.0
    }

    relation_optimization = RelationOptimization(
        loss_weights=loss_weights,
        visual_path=args.output,
        expand_dis=args.expand_dis, 
        toleration_dis=args.toleration_dis,
        visual_frames=100
    )

    # inference relationships between objects from GT
    relation_optimization.generate_relation(scene)
    # relation_optimization.generate_relation(gt_scene)
    
    # visualize GT scene
    background = visualize_relation(scene, layout=True, relation=False)
    save_image(background, os.path.join(args.output, f'gt.png'))
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    background = (background * 0.5).astype(np.uint8)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    relation_optimization.visual_background = background

    # randomize scene bdb3d
    gt_data = collate_fn(scene.data)
    relation_optimization.randomize_scene(scene)

    # visualize randomized scene
    image = visualize_relation(scene, background, relation=False)
    save_image(image, os.path.join(args.output, f'diff.png'))

    # to tensor
    optim_data = collate_fn(scene.data)
    relation_label = relation_from_bins(optim_data, None)
    optim_data['objs'].update(relation_label['objs'])
    optim_data['relation'] = relation_label['relation']

    # run relation optimization with visualization
    optim_bdb3d = relation_optimization.optimize(optim_data, 
                                                 steps=100,
                                                 visual=True,
                                                 lr=1)
    # optim_data['objs']['bdb3d'].update(IGTransform(optim_data).campix2world(optim_bdb3d))

    # evaluate bdb3d with gt
    optim_bdb3d.update(IGTransform(optim_data).campix2world(optim_bdb3d))
    compare_bdb3d(optim_data['objs']['bdb3d'], gt_data['objs']['bdb3d'], 'to_gt_before: ')
    compare_bdb3d(optim_bdb3d, optim_data['objs']['bdb3d'], 'from_initial: ')
    compare_bdb3d(optim_bdb3d, gt_data['objs']['bdb3d'], 'to_gt: ')


if __name__ == "__main__":
    main()
