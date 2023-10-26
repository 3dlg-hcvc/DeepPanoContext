import numpy as np
import seaborn as sns
import scipy.io as sio
import os


RLSD32CLASSES = ['bathtub', 'bed', 'blinds', 'cabinet', 'chair', 'chest_of_drawers', 
                  'clothes dryer', 'counter', 'curtain', 'cushion', 'dishwasher', 'door', 
                  'lighting', 'microwave', 'mirror', 'oven', 'picture', 'plant', 
                  'refrigerator', 'seating', 'shelving', 'shower', 'sink', 'sofa', 
                  'stool', 'stove', 'table', 'toilet', 'towel', 'tv_monitor', 'washing machine', 'window']
RLSD16CLASSES_NOCARE = ['beam', 'board_panel', 'ceiling', 'clothes', 'column', 'fireplace', 
                    'floor', 'furniture', 'gym_equipment', 'misc', 'objects', 'railing', 
                    'stairs', 'unlabeled', 'void', 'wall']
RLSD48CLASSES = RLSD32CLASSES + RLSD16CLASSES_NOCARE

CUSTOM2RLSD = {'pictures': 'picture', 'platn': 'plant', 'light': 'lighting', 'painting': 'picture', 'ligth': 'lighting', 'toilet paper': 'toilet', 'door wat': 'door', 'shelf': 'shelving', 'paint': 'picture', 'monitor': 'tv_monitor', 'light stand': 'lighting', 'tv': 'tv_monitor', 'lamp': 'lighting', 'pentant light': 'lighting'} # grill, basket

common_objects = ['counter', 'picture', 'microwave', 'sink', 'oven', 'dishwasher', 'chair', 'mirror', 'shower', 'table', 'stove', 'stool', 'bed', 'plant', 'door', 'bathtub', 'sofa', 'cushion', 'toilet', 'window']
rlsd_common_mapping = {k:k for k in common_objects}
RLSD32_2_IG56 = {'washing machine': 'washer', 'cabinet': 'bottom_cabinet', 'tv_monitor': 'monitor', 'lighting': 'floor_lamp', 'seating': 'bench', 'refrigerator': 'fridge', 'chest_of_drawers': 'chest', 'clothes dryer': 'dryer', 'shelving': 'shelf'}
RLSD32_2_IG56.update(rlsd_common_mapping)

COMMON25CLASSES = ['bathtub', 'bed', 'cabinet', 'chair', 'chest_of_drawers', 'clothes dryer', 'counter', 'cushion', 'dishwasher', 'microwave', 'mirror', 'oven', 'picture', 'plant', 'refrigerator', 'shelving', 'shower', 'sink', 'sofa', 'stool', 'stove', 'table', 'toilet', 'tv_monitor', 'washing machine']


NYU40CLASSES = ['bag', 'bathtub', 'bed', 'blinds', 'books', 'bookshelf', 'box', 'cabinet', 'ceiling', 'chair', 'clothes', 'counter', 'curtain', 'desk', 'door', 'dresser', 'floor', 'floor mat', 'lamp', 'mirror', 'nightstand', 'otherfurniture', 'otherprop', 'otherstructure', 'paper', 'person', 'picture', 'pillow', 'refrigerator', 'shelves', 'shower curtain', 'sink', 'sofa', 'table', 'television', 'toilet', 'towel', 'unknown', 'wall', 'whiteboard', 'window']

NYU40_2_COMMON25 = {'bookshelf': 'shelving', 'desk': 'table', 'dresser': 'chest_of_drawers', 'floor mat': 'carpet',
                 'lamp': 'lighting', 'nightstand': 'chest_of_drawers', 'pillow': 'cushion', 'shelves': 'shelving',
                 'television': 'tv_monitor'}

NYU40_2_IG56 = {'bookshelf': 'shelf', 'cabinet': 'bottom_cabinet', 'desk': 'table',
                'dresser': 'counter', 'floor mat': 'carpet', 'lamp': 'floor_lamp', 'nightstand': 'chest',
                'pillow': 'cushion', 'refrigerator': 'fridge', 'shelves': 'shelf',  'television': 'monitor'}

NYU40_2_PSU45 = {'bookshelf': 'shelving', 'desk': 'table', 'dresser': 'chest_of_drawers', 'floor mat': 'carpet',
                 'lamp': 'lighting', 'nightstand': 'chest_of_drawers', 'pillow': 'cushion', 'shelves': 'shelving',
                 'shower curtain': 'curtain', 'television': 'tv_monitor'}


PSU45CLASSES = RLSD32CLASSES + ['basket', 'range_hood', 'carpet', 'piano', 'coffee_machine', 'clock', 'guitar', 'heater', 'laptop', 'speaker', 'towel_rack', 'trash_can', 'treadmill']

IG56_2_PSU45 = {'bench': 'seating', 'bottom_cabinet': 'cabinet', 'bottom_cabinet_no_top': 'cabinet',
           'chest': 'chest_of_drawers', 'coffee_table': 'table', 'console_table': 'table',
           'cooktop': 'stove', 'crib': 'bed', 'dryer': 'clothes dryer', 'floor_lamp': 'lighting',
           'fridge': 'refrigerator', 'grandfather_clock': 'clock', 'loudspeaker': 'speaker',
           'monitor': 'tv_monitor', 'office_chair': 'chair', 'pool_table': 'table', 'shelf': 'shelving',
           'sofa_chair': 'chair', 'speaker_system': 'speaker', 'standing_tv': 'tv_monitor', 'table_lamp': 'lighting', 
           'top_cabinet': 'cabinet', 'wall_clock': 'clock', 'wall_mounted_tv': 'tv_monitor', 'washer': 'washing machine'}

IG56CLASSES = [
    'basket', 'bathtub', 'bed', 'bench', 'bottom_cabinet',
    'bottom_cabinet_no_top', 'carpet', 'chair', 'chest',
    'coffee_machine', 'coffee_table', 'console_table',
    'cooktop', 'counter', 'crib', 'cushion', 'dishwasher',
    'door', 'dryer', 'fence', 'floor_lamp', 'fridge',
    'grandfather_clock', 'guitar', 'heater', 'laptop',
    'loudspeaker', 'microwave', 'mirror', 'monitor',
    'office_chair', 'oven', 'piano', 'picture', 'plant',
    'pool_table', 'range_hood', 'shelf', 'shower', 'sink',
    'sofa', 'sofa_chair', 'speaker_system', 'standing_tv',
    'stool', 'stove', 'table', 'table_lamp', 'toilet',
    'top_cabinet', 'towel_rack', 'trash_can', 'treadmill',
    'wall_clock', 'wall_mounted_tv', 'washer', 'window'
]

IG59CLASSES = IG56CLASSES + ['walls', 'floors', 'ceilings']

WIMR11CLASSES = [
    'bed', 'painting', 'table', 'mirror', 'window', 'chair',
    'sofa', 'door', 'cabinet', 'bedside', 'tv',
]

WIMR2PC = {
    'bed': [
        'bed',
        'bed:outside',
        'bed:outside room',
        'bed:outside room ',
        'outside room bed',
        'baby bed'
    ],
    'painting': [
        'painting',
        'paitning',
        'paint',
        'picture',
        'picture: inside',
        'outside room picture',
        'picture:outside room',
        'picture: outside',
        'picture: outside room',
        'photo',
        'poster'
    ],
    'table': [
        'table',
        'table:outside room ',
        'table: outside room',
        'table:outside room',
        'outside room table',
        'round table',
        'round table:outside',
        'dressing table',
        'desk',
        'desk:outside',
        'desk:outside room',
        'dining table',
        'dining table ',
        'dining table:outside ',
        'dining table:outside',
        'dining table: outside',
        'outside dining table',
        'console table',
        'console table ',
        'console table',
        'console table:outside',
        'coffee table',
        'coffee table:outside',
        'end table',
        'bar table',
        'bar table:outside room ',
        'kitchen table',
        'desk and chair'
    ],
    'mirror': [
        'mirror',
        'mirror:outside room',
        'outside room mirror'
    ],
    'window': [
        'window',
        'window:outside',
        'window:outside room',
        'window:outside room ',
        'window: outside room',
        'outside room window'
    ],
    'chair': [
        'chair',
        'chair:outside',
        'chair: outside',
        'chair:outside room',
        'chair:outside room ',
        'chair: outside room',
        'outside room chair',
        'deck chair',
        'deck chair:outside room',
        'chair and table'
    ],
    'sofa': [
        'sofa',
        'sofa:outside',
        'sofa:outside room',
        'sofa:outside room ',
        'outside room sofa'
    ],
    'door': [
        'door',
        'doorway',
        'door non-4pt-polygon'
    ],
    'cabinet': [
        'cabinet',
        'cabinet:outside',
        'cabinet:outside room',
        'cabinet: outside room',
        ' cabinet:outside room',
        'outside room cabinet',
        'wardrobe',
        'wardrobe:outside'
    ],
    'bedside': [
        'bedside',
        'beside',
        'outside room nightstand',
        'nightstand',
        'nightstand:outside'
    ],
    'tv': [
        'tv',
        'TV',
        'tv:outside room ',
        'TV set'
    ],
}

PC2WIMR = {s: t for t, sl in WIMR2PC.items() for s in sl}

PC12CLASSES = [
    'bed', 'painting', 'nightstand', 'window', 'mirror', 'desk',
    'wardrobe', 'tv', 'door', 'chair', 'sofa', 'cabinet'
]

IG2PC = {
    'bed': 'bed', 'crib': 'bed', 
    'picture': 'painting', 
    'chest': 'nightstand', 
    'window': 'window', 
    'mirror': 'mirror', 
    'table': 'desk', 'coffee_table': 'desk', 'console_table': 'desk', 'pool_table': 'desk',
    'shelf': 'wardrobe',
    'standing_tv': 'tv', 'monitor': 'tv', 'wall_mounted_tv': 'tv', 
    'door': 'door', 
    'chair': 'chair', 'office_chair': 'chair', 'sofa_chair': 'chair', 'stool': 'chair', 
    'sofa': 'sofa', 
    'bottom_cabinet': 'cabinet', 'bottom_cabinet_no_top': 'cabinet', # 'top_cabinet': 'cabinet'
}

PC2IG = {
    'bed': 'bed', 
    'painting': 'picture',
    'table': 'table',
    'mirror': 'mirror',
    'window': 'window',
    'chair': 'chair',
    'sofa': 'sofa',
    'door': 'door',
    'cabinet': 'bottom_cabinet',
    'bedside': 'bottom_cabinet',
    'tv': 'standing_tv',
    'shelf': 'shelf'
}

colorbox_path = 'external/cooperative_scene_parsing/evaluation/vis/igibson_colorbox.mat'
igibson_colorbox = np.array(sns.hls_palette(n_colors=len(IG59CLASSES), l=.45, s=.8))
rlsd_cls45_colorbox = np.array(sns.hls_palette(n_colors=len(PSU45CLASSES), l=.45, s=.8))
rlsd_cls25_colorbox = np.array(sns.hls_palette(n_colors=len(COMMON25CLASSES), l=.45, s=.8))
if not os.path.exists(colorbox_path):
    sio.savemat(colorbox_path, {'igibson_colorbox': igibson_colorbox})

metadata = {}


def get_dataset_name(split):
    if split.endswith(('.json', '/')):
        name = split.split('/')[-2]
    else:
        name = os.path.basename(split)
    return name


def generate_bins(b_min, b_max, n):
    bins_width = (b_max - b_min) / n
    bins = np.arange(b_min, b_max, bins_width).astype(np.float32)
    bins = np.stack([bins, bins + bins_width]).T
    return bins


def bins_config():
    metadata['dis_bins'] = generate_bins(0, 12, 6)
    metadata['ori_bins'] = generate_bins(-np.pi, np.pi, 6)
    metadata['rot_bins'] = generate_bins(-np.pi, np.pi, 8)[:, 0]
    metadata['lat_bins'] = generate_bins(-np.pi / 2, np.pi / 2, 6)
    metadata['lon_bins'] = generate_bins(-np.pi, np.pi, 12)
    # metadata['pitch_bins'] = generate_bins(np.deg2rad(-20), np.deg2rad(60), 2)
    metadata['pitch_bins'] = generate_bins(np.deg2rad(10), np.deg2rad(30), 2)
    # metadata['roll_bins'] = generate_bins(np.deg2rad(-20), np.deg2rad(20), 2)
    metadata['roll_bins'] = generate_bins(np.deg2rad(-10), np.deg2rad(10), 2)
    metadata['layout_ori_bins'] = generate_bins(np.deg2rad(-45), np.deg2rad(45), 2)


bins_config()
