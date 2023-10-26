import os
import json
# import random
import pickle
from tqdm import tqdm

# random.seed(123)

ig_train_json = "/project/3dlg-hcvc/rlsd/data/psu/igibson_cls25/train.json"
ig_test_json = "/project/3dlg-hcvc/rlsd/data/psu/igibson_cls25/test.json"
s3d_train_json = "/project/3dlg-hcvc/rlsd/data/psu/s3d_cls25/train.json"
s3d_test_json = "/project/3dlg-hcvc/rlsd/data/psu/s3d_cls25/test.json"

data_dir = "/project/3dlg-hcvc/rlsd/data/psu/s3d_ig_cls25"

ig_train = json.load(open(ig_train_json))
ig_test = json.load(open(ig_test_json))
s3d_train = json.load(open(s3d_train_json))
s3d_test = json.load(open(s3d_test_json))

for s3d_pkl_file in tqdm(s3d_train):
    data = pickle.load(open(os.path.join(data_dir, s3d_pkl_file), 'rb'))
    img_dict = data['image_path'].copy()
    for k, v in data['image_path'].items():
        img_dict[k].replace('s3d_cls25', 's3d_ig_cls25')
    data['image_path'] = img_dict
    with open(os.path.join(data_dir, s3d_pkl_file), 'wb') as f:
        pickle.dump(data, f)

mix_train = ig_train + s3d_train
mix_test = ig_test + s3d_test

with open(os.path.join(data_dir, 'train.json'), 'w') as f:
    json.dump(mix_train, f, indent=4)
    
with open(os.path.join(data_dir, 'test.json'), 'w') as f:
    json.dump(mix_test, f, indent=4)

