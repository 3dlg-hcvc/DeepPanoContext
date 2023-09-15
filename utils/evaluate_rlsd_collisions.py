import os
import json
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from copy import deepcopy

from utils.igibson_utils import IGScene
from models.pano3d.dataloader import IGSceneDataset
from utils.visualize_utils import IGVisualizer
from utils.relation_utils import RelationOptimization, relation_from_bins
from utils.basic_utils import read_pkl
from utils.mesh_utils import MeshIO, write_ply_rgb_face, create_layout_mesh, create_bdb3d_mesh
from models.eval_metrics import bdb3d_iou, bdb2d_iou, classification_metric, AverageMeter, \
    BinaryClassificationMeter, ClassMeanMeter
from configs import data_config


data_dir = "/project/3dlg-hcvc/rlsd/data/psu/rlsd_real"
data_paths = sorted(glob(os.path.join(data_dir, '*', '*', '*', 'data.pkl')))
# scene_names = [f for f in os.listdir(data_dir) if not f.endswith(".json")]

num_cls = len(data_config.RLSD32CLASSES)
cls_num = np.zeros(num_cls, dtype=np.int)
cls_cls_tch_num = np.zeros([num_cls, num_cls], dtype=np.int)
cls_cls_tch = np.zeros([num_cls, num_cls])


def merge_layout_bdb3d_mesh(data, objs, colorbox=None, separate=False, camera_color=None, layout_color=None, texture=False, filename=None):
    mesh_io = MeshIO()
    
    for k, obj in enumerate(objs):
        bdb3d = obj['bdb3d']
        mesh_world = create_bdb3d_mesh(bdb3d, radius=0.015)
        mesh_io[k] = mesh_world

    # add layout mesh
    if layout_color is not None:
        layout_mesh = create_layout_mesh(data, color=layout_color, texture=texture)
        if layout_mesh is not None:
            mesh_io['layout_mesh'] = layout_mesh
    
    all_verts, all_faces, all_colors = [], [], []
    for k, m in mesh_io.items():
        if k == 'camera': color = camera_color
        elif k == 'layout_mesh': color = layout_color
        else: 
            color = colorbox[objs[k]['label']]
        cur_num_verts = len(all_verts)
        all_verts.extend(m.vertices.tolist())
        all_faces.extend((m.faces + cur_num_verts).tolist())
        all_colors.extend([color] * len(m.vertices))
    
    if filename is not None:
        write_ply_rgb_face(np.array(all_verts), np.array(all_colors), np.array(all_faces), filename)

    if separate:
        return mesh_io

    return mesh_io.merge()


def evaluate_collision(data, arch_id, metric, metric_archs):
    # generate relation between estimated objects and layout
    # 0.1m of toleration distance when measuring collision
    relation_optimization = RelationOptimization(expand_dis=-0.1)
    data['objs'] = deepcopy(data['objs'])
    rel_scene = IGScene(data)
    relation_optimization.generate_relation(rel_scene)
    relation = rel_scene['relation']
    
    global cls_cls_tch, cls_cls_tch_num, cls_num
    cls_cls_tch_num += relation['cls_cls_tch_num']
    cls_cls_tch += relation['cls_cls_tch']
    cls_num += relation['cls_num']

    # collision metrics
    metric_archs[arch_id]['collision_pairs'].append(relation['obj_obj_tch'].sum() / 2)
    metric_archs[arch_id]['collision_objs'].append(relation['obj_obj_tch'].any(axis=0).sum())
    metric_archs[arch_id]['collision_walls'].append(relation['obj_wall_tch'].any(axis=-1).sum())
    metric_archs[arch_id]['collision_ceil'].append(sum(o['ceil_tch'] for o in rel_scene['objs']))
    metric_archs[arch_id]['collision_floor'].append(sum(o['floor_tch'] for o in rel_scene['objs']))
    
    metric['collision_pairs'].append(relation['obj_obj_tch'].sum() / 2)
    metric['collision_objs'].append(relation['obj_obj_tch'].any(axis=0).sum())
    metric['collision_walls'].append(relation['obj_wall_tch'].any(axis=-1).sum())
    metric['collision_ceil'].append(sum(o['ceil_tch'] for o in rel_scene['objs']))
    metric['collision_floor'].append(sum(o['floor_tch'] for o in rel_scene['objs']))
    
    # # save reconstructed relations for relation fidelity evaluation
    # relation_optimization = RelationOptimization(expand_dis=cfg.config['data'].get('expand_dis', 0.1))
    # relation_optimization.generate_relation(est_rel_scene)
    # est_rel_scene['relation'] = relation_from_bins(relation, None)
    # rel_scenes.append(est_rel_scene)

metric = defaultdict(AverageMeter)
metric_archs = defaultdict(lambda: ClassMeanMeter(AverageMeter))
for pkl_file in tqdm(data_paths):
    # scene_dir = os.path.join(out_dir, scene_name)
    # os.makedirs(scene_dir, exist_ok=True)
    arch_id, pano_id, task_id = pkl_file.split('/')[-4:-1]
    # scene_polygons = dict()
    # scene_rooms = defaultdict(list)
    # scene_layouts = defaultdict(dict)
    # scene_objs = defaultdict(dict)
    # for i in range(100):
    # pkl_file = os.path.join(data_dir, scene_name, camera_id, 'data.pkl')
    data = read_pkl(pkl_file)
    # room_name = data['room']
    # scene_rooms[room_name].append(camera_id)
    # if room_name not in scene_polygons:
    #     scene_layouts[room_name]["layout"] = data["layout"]
    #     corners = data["layout"]["manhattan_world"]
    #     room = Polygon(corners[:(len(corners)//2), :2])
    #     scene_polygons[room_name] = room
    # scene_objs[room_name].update({obj['id']: obj for obj in data['objs'] if obj['id'] not in scene_objs[room_name]})
    evaluate_collision(data.copy(), arch_id, metric, metric_archs)
    
    # plot_ig_rooms_layout(scene_polygons,
    #                      os.path.join(scene_dir, "rooms.png"))
    # with open(os.path.join(scene_dir, "cameras.json"), "w") as f:
    #     json.dump(scene_rooms, f, indent=4)
        
for arch_id in metric_archs:
    for k in metric_archs[arch_id]:
        metric_archs[arch_id][k] = metric_archs[arch_id][k]()
for k in metric:
    metric[k] = metric[k]()

metric.update(metric_archs)
with open(os.path.join(data_dir, "collisions.json"), "w") as f:
    json.dump(metric, f, indent=4)
    
plt.savefig('num.png')
_, ax = plt.subplots(figsize=(14,10))
ax = sns.barplot(x=data_config.RLSD32CLASSES, y=cls_num)
plt.xticks(rotation=60)
plt.savefig('num.png')

_, ax1 = plt.subplots(figsize=(14,10))
ax1 = sns.heatmap(cls_cls_tch_num, annot=True, fmt='d', xticklabels=data_config.RLSD32CLASSES, yticklabels=data_config.RLSD32CLASSES)
plt.savefig('heat.png')

_, ax2 = plt.subplots(figsize=(14,10))
ax2 = sns.heatmap(cls_cls_tch/(cls_cls_tch_num+1), annot=True, fmt='.2f', xticklabels=data_config.RLSD32CLASSES, yticklabels=data_config.RLSD32CLASSES)
plt.savefig('prob.png')

# import pdb; pdb.set_trace()