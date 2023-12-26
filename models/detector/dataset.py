from detectron2.utils.logger import setup_logger
setup_logger()

import random, argparse
from tqdm import tqdm
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg as default_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from configs.data_config import IG56CLASSES, WIMR11CLASSES, PC12CLASSES, get_dataset_name, R3DS32CLASSES, COMMON25CLASSES
from utils.visualize_utils import detectron_gt_sample, visualize_igibson_detectron_gt
from utils.image_utils import show_image
from models.pano3d.dataloader import SceneDataset


def register_detection_dataset(path, real=None):
    dataset = get_dataset_name(path)
    for d in ["train" , "test"]:
        DatasetCatalog.register(
            f"{dataset}_{d}", lambda d=d: get_dataset_dicts(path, d))
        if dataset.startswith(('igibson',)):
            thing_classes = IG56CLASSES
        elif dataset.startswith(('r3ds', 's3d', 'ig_rr')):
            if 'ig' in dataset:
                thing_classes = IG56CLASSES
            else:
                thing_classes = R3DS32CLASSES
        elif dataset.startswith(('pano_context', 'wimr')) or real == True:
            thing_classes = WIMR11CLASSES
        else:
            thing_classes = IG56CLASSES
            # raise NotImplementedError
        if '25' in dataset:
            thing_classes = COMMON25CLASSES
        MetadataCatalog.get(f"{dataset}_{d}").set(thing_classes=thing_classes)


def get_dataset_dicts(folder, mode):
    dataset_name = get_dataset_name(folder)
    if dataset_name.startswith(('igibson', 'r3ds', 'ig_rr', 's3d')):
        dataset = SceneDataset({'data': {'split': folder}}, mode)
    else:
        raise NotImplementedError
    dataset_dicts = []

    for idx in tqdm(range(len(dataset)), desc=f'Loading {dataset_name}'):
        scene = dataset.get_scene(idx)
        record = detectron_gt_sample(scene, idx)
        record['scene'] = scene
        dataset_dicts.append(record)

    return dataset_dicts


def get_cfg(dataset='igibson', config='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'):
    dataset = get_dataset_name(dataset)
    cfg = default_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.DATASETS.TRAIN = (f"{dataset}_train",)
    cfg.DATASETS.TEST = (f"{dataset}_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)  # Let training initialize from model zoo
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    num_classes = len(MetadataCatalog.get(f"{dataset}_train").get('thing_classes'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.FORMAT = 'RGB'
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson detection GT.')
    parser.add_argument('--dataset', type=str, default='data/igibson',
                        help='The path of the dataset')
    args = parser.parse_args()

    register_detection_dataset(args.dataset)
    dataset_dicts = DatasetCatalog.get(f"{get_dataset_name(args.dataset)}_train")
    for sample in random.sample(dataset_dicts, 3):
        image = visualize_igibson_detectron_gt(sample)
        show_image(image)
    return

if __name__ == "__main__":
    main()
