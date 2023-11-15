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
from utils.relation_utils import test_bdb3ds, RelationOptimization, relation_from_bins
from utils.basic_utils import read_pkl
from utils.mesh_utils import MeshIO, write_ply_rgb_face, create_layout_mesh, create_bdb3d_mesh
from utils.transform_utils import num2bins, bdb3d_corners, expand_bdb3d
from utils.layout_utils import wall_bdb3d_from_manhattan_world_layout, manhattan_world_layout_info
from models.eval_metrics import AverageMeter, BinaryClassificationMeter, ClassMeanMeter


data_dir = "/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25"
data_paths = sorted(glob(os.path.join(data_dir, '*', '*', '*', 'data.pkl')))

gt_ro = RelationOptimization(expand_dis=0.1, use_anno_supp=True)
est_ro = RelationOptimization(expand_dis=0.1)

metric_rels = defaultdict(BinaryClassificationMeter)

for pkl_file in tqdm(data_paths):
    arch_id, pano_id, task_id = pkl_file.split('/')[-4:-1]
    gt_scene = IGScene.from_pickle(pkl_file)
    est_scene = IGScene.from_pickle(pkl_file)
    if "layout" not in gt_scene.data:
        continue

    gt_ro.generate_relation(gt_scene)
    est_ro.generate_relation(est_scene)
    # if np.any(est_scene['relation']['obj_obj_col']):
    #     import pdb; pdb.set_trace()

    # gt_rels = relation_from_bins(gt_scene.data, None)['relation']
    # est_rels = relation_from_bins(est_scene.data, None)['relation']
    gt_rel = gt_scene['relation']
    est_rel = est_scene['relation']

    # id_gt = np.array([obj['gt'] for obj in est_scene['objs']])
    # id_match = id_gt >= 0
    # id_gt = id_gt[id_match]

    if len(gt_scene['objs']) <= 0:
        continue

    for k in ['floor_supp', 'ceil_supp']:
        est_v = np.array([o[k] for o in est_scene['objs']])#[id_match]
        gt_v = np.array([o[k] for o in gt_scene['objs']])#[id_gt]
        metric_rels[f"{k}_classify"].add({'est': est_v, 'gt': gt_v})

    for k in ['obj_obj_supp', 'obj_wall_supp']:
        # match estimated objects with ground truth
        if 'relation' not in est_scene.data:
            continue
        est_v = est_scene['relation'][k]
        # est_v = est_v[id_match]
        gt_v = gt_rel[k]#[id_gt]
        # if k.startswith('obj_obj'):
        #     if len(id_gt) <= 1:
        #         continue
        #     mask = ~np.eye(len(id_gt), dtype=np.bool)
        #     est_v = est_v[:, id_match][mask]
        #     gt_v = gt_v[:, id_gt][mask]

        est_v = est_v.reshape(-1)
        gt_v = gt_v.reshape(-1)
        if est_v.dtype == np.bool:
            metric_rels[f"{k}_classify"].add({'est': est_v, 'gt': gt_v})

print(', '.join([f"{k}: {v}" for k, v in metric_rels.items()]))
