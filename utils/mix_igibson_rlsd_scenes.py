import os
import json
import random
import pickle
from tqdm import tqdm

random.seed(123)

ig_train_json = "/project/3dlg-hcvc/rlsd/data/psu/igibson_cls25/train.json"
# ig_test_json = "/project/3dlg-hcvc/rlsd/data/psu/igibson_cls25/test.json"
rlsd_train_json = "/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25/train.json"
# rlsd_test_json = "/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25/test.json"

data_dir = "/project/3dlg-hcvc/rlsd/data/psu/ig_rr_cls25"

ig_train = json.load(open(ig_train_json))
# ig_test = json.load(open(ig_test_json))
rlsd_train = json.load(open(rlsd_train_json))
# rlsd_test = json.load(open(rlsd_test_json))

mix_train, mix_test = [], []
mix_train.extend(ig_train[::2]) # 500

rlsd_train_sample = sorted(random.sample(rlsd_train, 500))
mix_train.extend(rlsd_train_sample) # 500
for rlsd_pkl_file in tqdm(rlsd_train_sample):
    data = pickle.load(open(os.path.join(data_dir, rlsd_pkl_file), 'rb'))
    img_dict = data['image_path'].copy()
    for k, v in data['image_path'].items():
        img_dict[k].replace('rlsd_real_cls25', 'ig_rr_cls25')
    data['image_path'] = img_dict
    with open(os.path.join(data_dir, rlsd_pkl_file), 'wb') as f:
        pickle.dump(data, f)
        
with open(os.path.join(data_dir, 'train.json'), 'w') as f:
    json.dump(mix_train, f)