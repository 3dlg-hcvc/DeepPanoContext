import os
import pandas as pd
from PIL import Image
from .igibson_utils import hash_split


data_dirs = {
    "real": {
        "rgb": "/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_rgb_panos",
        "inst": "/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_instance_panos"
    },
    "syn": {
        "rgb": "/project/3dlg-hcvc/rlsd/data/annotations/equirectangular_objects_arch",
        "inst": "/project/3dlg-hcvc/rlsd/data/annotations/equirectangular_instance"
    }
}


def encode_rgba(arr):
    arr = arr[:,:,0]*(pow(2,24)) + arr[:,:,1]*(pow(2,16)) + arr[:,:,2]*(pow(2,8)) + arr[:,:,3]
    return arr


def prepare_images(args):
    if os.path.exists(os.path.join(args.output, args.scene_name, "rgb.png")):
        return
    house_id = args.scene_name.split("_")[0]
    pano_id = args.scene_name.split("/")[-1]
    task_id = args.task_id
    if args.img_mode == "real":
        rgb_path = f"{data_dirs[args.img_mode]['rgb']}/{house_id}/{pano_id}.png"
        inst_path = f"{data_dirs[args.img_mode]['inst']}/{house_id}/{pano_id}.objectId.encoded.png"
    elif args.img_mode == "syn":
        rgb_path = f"{data_dirs[args.img_mode]['rgb']}/{task_id}/{pano_id}.png"
        inst_path = f"{data_dirs[args.img_mode]['inst']}/{task_id}/{pano_id}.objectId.encoded.png"
    else:
        raise NotImplemented
    Image.open(rgb_path).convert("RGB").resize((1024, 512)).save(os.path.join(args.output, args.scene_name, "rgb.png"))
    Image.open(inst_path).resize((1024, 512), Image.NEAREST).save(os.path.join(args.output, args.scene_name, "seg.png"))


def create_data_splits(args, data_paths):
    split = {'train': [], 'test': []}
    scenes = {'train': set(), 'test': set()}
    if args.split_by == 'house':
        mp3d_splits = ['train', 'val', 'test']
        house2split = {}
        for sp in mp3d_splits:
            rlsd_sp = sp if sp != 'val' else 'train'
            split_file = f"/project/3dlg-hcvc/rlsd/data/mp3d/split/{sp}.txt"
            houses = [l.strip() for l in open(split_file)]
            house2split.update({h: rlsd_sp for h in houses})
        for camera in data_paths:
            if camera is None: continue
            arch_id = camera.split('/')[-4] # based on arch id
            house_id, _ = arch_id.split('_')
            sp = house2split[house_id]
            path = os.path.join(*camera.split('/')[-4:])
            if sp == 'train':
                split['train'].append(path)
                scenes['train'].add(house_id)
            else:
                split['test'].append(path)
                scenes['test'].add(house_id)
    elif args.split_by == 'region':
        pass
        pano_df = pd.read_csv("/project/3dlg-hcvc/rlsd/data/annotations/pano_arch_element.csv")
        for camera in data_paths:
            if camera is None: continue
            arch_id, pano_id = camera.split('/')[-4:-2] # based on arch id
            region_id = pano_df[pano_df.pano_id == pano_id].iloc[0].new_region_id
            arch_region_id = f"{arch_id}_{region_id}"
            is_train = hash_split(args.train, arch_region_id)
            path = os.path.join(*camera.split('/')[-4:])
            if is_train:
                split['train'].append(path)
                scenes['train'].add(arch_region_id)
            else:
                split['test'].append(path)
                scenes['test'].add(arch_region_id)
    elif args.split_by == 'pano':
        for camera in data_paths:
            if camera is None: continue
            pano_id = camera.split('/')[-3] # based on panorama id
            is_train = hash_split(args.train, pano_id)
            path = os.path.join(*camera.split('/')[-4:])
            if is_train:
                split['train'].append(path)
                scenes['train'].add(pano_id)
            else:
                split['test'].append(path)
                scenes['test'].add(pano_id)
    
    return split, scenes