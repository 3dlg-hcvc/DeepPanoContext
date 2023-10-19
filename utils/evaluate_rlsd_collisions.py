import os
import json
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon, Point, MultiPolygon
from copy import deepcopy

from configs import data_config
from utils.igibson_utils import IGScene
from utils.relation_utils import test_bdb3ds
from utils.basic_utils import read_pkl
from utils.mesh_utils import MeshIO, write_ply_rgb_face, create_layout_mesh, create_bdb3d_mesh
from utils.transform_utils import num2bins, bdb3d_corners, expand_bdb3d
from utils.layout_utils import wall_bdb3d_from_manhattan_world_layout, manhattan_world_layout_info
from models.eval_metrics import AverageMeter, BinaryClassificationMeter, ClassMeanMeter


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


def generate_relation(scene, expand_dis):
        objs = scene['objs']
        n_objs = len(objs)
        obj_obj_rot = np.zeros([n_objs, n_objs], dtype=np.int)  # angles of clockwise rotation from a to b
        obj_obj_dis = np.zeros_like(obj_obj_rot, dtype=np.bool)  # is a further than b
        obj_obj_tch = obj_obj_dis.copy()  # is a touching b
        obj_obj_col = obj_obj_dis.copy()  # is a colliding b
        obj_obj_supp = obj_obj_dis.copy() # is a supported by b
        
        num_cls = len(data_config.RLSD32CLASSES)
        cls_num = np.zeros(num_cls, dtype=np.int)
        cls_cls_tch_num = np.zeros([num_cls, num_cls], dtype=np.int)
        cls_cls_tch = np.zeros([num_cls, num_cls])

        # object - object relationships
        for obj in objs:
            obj['bdb3d'].update(scene.transform.world2campix(obj['bdb3d']))

        for i_a, obj_a in enumerate(objs):
            obj_a_label = obj_a['label'][0]
            cls_num[obj_a_label] += 1
            if 'obj_parent' in obj_a and obj_a['obj_parent'] != -1:
                obj_obj_supp[i_a, obj_a['obj_parent']] = 1
            for i_b, obj_b in enumerate(objs):
                if i_a == i_b:
                    continue
                bdb3d_a = obj_a['bdb3d']
                bdb3d_b = obj_b['bdb3d']
                rot = np.mod(bdb3d_b['ori'] - bdb3d_a['ori'], np.pi * 2)
                rot = rot - np.pi * 2 if rot > np.pi else rot
                obj_obj_rot[i_a, i_b] = num2bins(data_config.metadata['rot_bins'], rot)
                obj_obj_dis[i_a, i_b] = bdb3d_a['dis'] > bdb3d_b['dis']
                obj_obj_tch[i_a, i_b] = bool(test_bdb3ds(bdb3d_a, bdb3d_b, - expand_dis))
                obj_obj_col[i_a, i_b] = bool(test_bdb3ds(bdb3d_a, bdb3d_b, expand_dis))
                
                if obj_obj_tch[i_a, i_b]:
                    from models.eval_metrics import bdb3d_iou
                    obj_a_label = obj_a['label'][0]
                    obj_b_label = obj_b['label'][0]
                    cls_cls_tch_num[obj_a_label, obj_b_label] += 1
                    corners_a, corners_b = bdb3d_corners(bdb3d_a), bdb3d_corners(bdb3d_b)
                    vol_a, obj_3d_iou = bdb3d_iou(corners_a, corners_b, union=False)
                    if vol_a == 0.:
                        continue
                    cls_cls_tch[obj_a_label, obj_b_label] += obj_3d_iou

        # object - floor/ceiling relationships
        layout = scene['layout']['manhattan_world']
        layout_info = manhattan_world_layout_info(layout)
        for obj in objs:
            bdb3d_shrink = expand_bdb3d(obj['bdb3d'], - expand_dis)
            corners_shrink = bdb3d_corners(bdb3d_shrink)
            bdb3d_expand = expand_bdb3d(obj['bdb3d'], expand_dis)
            corners_expand = bdb3d_corners(bdb3d_expand)
            corners_2d = corners_expand[:4, :2]
            obj['in_room'] = any(layout_info['layout_poly'].contains(Point(c)) for c in corners_2d)
            
            obj['floor_col'] = corners_shrink[:, -1].min() < layout_info['floor'] if obj['in_room'] else 0
            obj['ceil_col'] = corners_shrink[:, -1].max() > layout_info['ceil'] if obj['in_room'] else 0
            obj['floor_tch'] = corners_expand[:, -1].min() < layout_info['floor'] if obj['in_room'] else 0
            obj['ceil_tch'] = corners_expand[:, -1].max() > layout_info['ceil'] if obj['in_room'] else 0
            
            obj['floor_supp'] = 1
            obj['ceil_supp'] = 0

        walls_bdb3d = wall_bdb3d_from_manhattan_world_layout(layout)

        # object - wall relationships
        # get contour from layout estimation
        walls = []
        for wall_bdb3d in walls_bdb3d:
            wall_bdb3d['bdb3d'].update(scene.transform.world2campix(wall_bdb3d['bdb3d']))
            walls.append(wall_bdb3d)

        obj_wall_rot = np.zeros(
            [n_objs, len(walls)], dtype=np.int)  # angles of clockwise rotation from object to wall
        obj_wall_tch = np.zeros_like(obj_wall_rot, dtype=np.bool)  # is obj touching wall
        obj_wall_col = np.zeros_like(obj_wall_rot, dtype=np.bool)  # is obj colliding wall
        obj_wall_supp = obj_wall_tch.copy() # is obj supported by wall
        for i_obj, obj in enumerate(objs):
            if obj['wall_supp']:
                obj_wall_supp[i_obj, obj['wall_parent']] = 1
            for i_wall, wall in enumerate(walls):
                bdb3d_obj = obj['bdb3d']
                bdb3d_wall = wall['bdb3d']
                rot = np.mod(bdb3d_wall['ori'] - bdb3d_obj['ori'], np.pi * 2)
                rot = rot - np.pi * 2 if rot > np.pi else rot
                obj_wall_rot[i_obj, i_wall] = num2bins(data_config.metadata['rot_bins'], rot)
                obj_wall_tch[i_obj, i_wall] = test_bdb3ds(obj['bdb3d'], wall['bdb3d'], - expand_dis) if obj['in_room'] else 0
                obj_wall_col[i_obj, i_wall] = test_bdb3ds(obj['bdb3d'], wall['bdb3d'], expand_dis) if obj['in_room'] else 0
        
        # write to scene data
        if 'walls' in scene.data:
            for wall_old, wall_new in zip(scene['walls'], walls):
                wall_old.update(wall_new)
        else:
            scene['walls'] = walls
        scene['relation'] = {
            'obj_obj_rot': obj_obj_rot,
            'obj_obj_dis': obj_obj_dis,
            'obj_obj_tch': obj_obj_tch,
            'obj_obj_col': obj_obj_col,
            'obj_wall_rot': obj_wall_rot,
            'obj_wall_tch': obj_wall_tch,
            'obj_wall_col': obj_wall_col,
            'obj_obj_supp': obj_obj_supp,
            'obj_wall_supp': obj_wall_supp,
            'cls_cls_tch_num': cls_cls_tch_num,
            'cls_cls_tch': cls_cls_tch,
            'cls_num': cls_num
        }


def evaluate_collision(data, arch_id, metric, metric_archs):
    # generate relation between estimated objects and layout
    # 0.1m of toleration distance when measuring collision
    data['objs'] = deepcopy(data['objs'])
    rel_scene = IGScene(data)
    generate_relation(rel_scene, expand_dis=-0.1)
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
    

metric = defaultdict(AverageMeter)
metric_archs = defaultdict(lambda: ClassMeanMeter(AverageMeter))
for pkl_file in tqdm(data_paths):
    arch_id, pano_id, task_id = pkl_file.split('/')[-4:-1]
    data = read_pkl(pkl_file)
    evaluate_collision(data.copy(), arch_id, metric, metric_archs)
    
        
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

avg_iou = cls_cls_tch/(cls_cls_tch_num+1)
_, ax2 = plt.subplots(figsize=(14,10))
ax2 = sns.heatmap(avg_iou, annot=True, fmt='.2f', xticklabels=data_config.RLSD32CLASSES, yticklabels=data_config.RLSD32CLASSES)
plt.savefig('avg_iou.png')
np.save("avg_iou.npy", avg_iou)

col_prob = cls_cls_tch_num / (cls_num[None, ...]+1)
_, ax3 = plt.subplots(figsize=(14,10))
ax3 = sns.heatmap(col_prob, annot=True, fmt='.2f', xticklabels=data_config.RLSD32CLASSES, yticklabels=data_config.RLSD32CLASSES)
plt.savefig('col_prob.png')
np.save("col_prob.npy", col_prob)

weighted_col_prob = col_prob * avg_iou
_, ax4 = plt.subplots(figsize=(14,10))
ax4 = sns.heatmap(weighted_col_prob, annot=True, fmt='.2f', xticklabels=data_config.RLSD32CLASSES, yticklabels=data_config.RLSD32CLASSES)
plt.savefig('w_col_prob.png')
np.save("w_col_prob.npy", weighted_col_prob)

# import pdb; pdb.set_trace()