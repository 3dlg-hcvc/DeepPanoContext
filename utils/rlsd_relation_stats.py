import os
import json
import pickle
from glob import glob
from tqdm import tqdm

output = "/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25"
data_paths = sorted(glob(os.path.join(output, '*', '*', '*', 'data.pkl')))

relation = {
    'wall': {},
    'floor': {},
    'ceil': {}
}

data_w_mirror = set()

for dp in tqdm(data_paths):
    data = pickle.load(open(dp, 'rb'))
    for obj in data['objs']:
        if obj['floor_supp']:
            count = relation['floor'].get(obj['classname'][0], 0) + 1
            relation['floor'][obj['classname'][0]] = count 
        if obj['ceil_supp']:
            count = relation['ceil'].get(obj['classname'][0], 0) + 1
            relation['ceil'][obj['classname'][0]] = count 
        if obj['wall_supp']:
            count = relation['wall'].get(obj['classname'][0], 0) + 1
            relation['wall'][obj['classname'][0]] = count
        if 'mirror' in obj['classname']:
            data_w_mirror.add(dp)

with open("/project/3dlg-hcvc/rlsd/data/annotations/psu_relation_stats.json", "w") as f:
    json.dump(relation, f, indent=4)

with open("/project/3dlg-hcvc/rlsd/data/annotations/psu_mirror_data.json", "w") as f:
    json.dump(list(data_w_mirror), f, indent=4)

# import pdb; pdb.set_trace()