import os
import json
import argparse
from multiprocessing import Pool
from tqdm import tqdm

from configs.data_config import rlsd_cls45_colorbox, rlsd_cls25_colorbox, igibson_colorbox
from models.detector.dataset import register_detection_dataset
from .igibson_utils import IGScene
from .image_utils import save_image
from .visualize_utils import IGVisualizer
from .render_layout_bdb3d import render_view
from .relation_utils import RelationOptimization, visualize_relation
from .mesh_utils import save_mesh


def visualize_camera(args):
    arch_id, pano_id = args.scene_name.split("/")
    scene_folder = os.path.join(args.dataset, args.scene_name) if args.scene_name is not None else args.dataset
    camera_folder = os.path.join(scene_folder, args.task_id)
    if not os.path.exists(os.path.join(camera_folder, 'data.pkl')): return
    scene = IGScene.from_pickle(camera_folder, load_rlsd_obj=True)
    
    if args.save_scene_mesh:
        scene_mesh = scene.merge_rlsd_mesh(
            colorbox=rlsd_cls25_colorbox * 255,
            separate=False,
            layout_color=(17, 207, 67),
            texture=False
        )
        if len(scene_mesh.vertices) > 0:
            save_mesh(scene_mesh, os.path.join(scene_folder, args.task_id, 'scene_mesh.obj'))
            save_mesh(scene_mesh, os.path.join(scene_folder, args.task_id, 'scene_mesh.glb'))
            render_view(os.path.join(scene_folder, args.task_id, 'scene_mesh.obj'),
                        os.path.join(scene_folder, args.task_id, 'scene_mesh.png'))
    
        bdb3d_mesh = scene.merge_layout_bdb3d_mesh(
                colorbox=rlsd_cls25_colorbox * 255,
                separate=False,
                layout_color=(17, 207, 67),
                texture=False,
                # filename=os.path.join(scene_folder, args.task_id, 'layout_bdb3d.ply')
            )
        if len(bdb3d_mesh.vertices) > 0:
            save_mesh(bdb3d_mesh, os.path.join(scene_folder, args.task_id, 'layout_bdb3d.obj'))
            render_view(os.path.join(scene_folder, args.task_id, 'layout_bdb3d.obj'),
                        os.path.join(scene_folder, args.task_id, 'layout_bdb3d.png'))
        
    if 'layout' in scene.data and 'objs' in scene.data and scene['objs'] and 'bdb3d' in scene['objs'][0]:
        ro = RelationOptimization(expand_dis=args.expand_dis, use_anno_supp=True)
        ro.generate_relation(scene)
        image = visualize_relation(scene, layout=True, relation=True, collision=True, colorbox=rlsd_cls25_colorbox)
        save_path = os.path.join(scene_folder, args.task_id, 'relation.png')
        save_image(image, save_path)
        as_image = visualize_relation(scene, layout=True, support=True, relation=False, colorbox=rlsd_cls25_colorbox)
        save_path = os.path.join(scene_folder, args.task_id, 'as.png')
        save_image(as_image, save_path)
        del scene.data['relation']
        ro = RelationOptimization(expand_dis=args.expand_dis, use_anno_supp=False)
        ro.generate_relation(scene)
        hs_image = visualize_relation(scene, layout=True, support=True, relation=False, colorbox=rlsd_cls25_colorbox)
        save_path = os.path.join(scene_folder, args.task_id, 'hs.png')
        save_image(hs_image, save_path)

    visualizer = IGVisualizer(scene, gpu_id=args.gpu_id, debug=args.debug)

    image = visualizer.image('rgb')
    image = visualizer.layout(image, total3d=False)
    image = visualizer.objs3d(image, bbox3d=True, axes=False, centroid=False, info=False, thickness=1)
    det_image = visualizer.holes(image)
    save_path = os.path.join(scene_folder, args.task_id, 'det3d.png')
    save_image(det_image, save_path)
    # image = visualizer.bfov(image, thickness=1, include=('walls', 'objs'))
    image = visualizer.bdb2d(image)
    save_path = os.path.join(scene_folder, args.task_id, 'visual.png')
    save_image(image, save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson scenes.')
    parser.add_argument('--dataset', type=str, default='/project/3dlg-hcvc/rlsd/data/psu/rlsd_real',
                        help='The path of the rlsd dataset')
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
    parser.add_argument('--save_scene_mesh', default=False, action='store_true',
                        help='')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    args = parser.parse_args()
    register_detection_dataset(args.dataset)
    
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

