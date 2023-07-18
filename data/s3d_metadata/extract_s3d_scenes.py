import os
import json
from glob import glob
from tqdm import tqdm

rooms = sorted(glob("/datasets/external/Structured3D/data/scene_*/2D_rendering/*"))

errata_scenes = []
errata_rooms = []
with open("/local-scratch/qiruiw/research/DeepPanoContext/data/s3d_metadata/errata.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            continue
        tokens = line.split("_")
        if len(tokens) == 2:
            errata_scenes.append(tokens[1])
        elif len(tokens) == 4:
            errata_rooms.append(f"{tokens[1]}_{tokens[3]}")
            
import pdb; pdb.set_trace()
            
valid_rooms = []
for room in tqdm(rooms):
    scene_name, _, room_id = room.split("/")[-3:]
    scene_id = scene_name.split("_")[1]
    if scene_id in errata_scenes:
        continue
    elif f"{scene_id}_{room_id}" in errata_rooms:
        continue
    else:
        valid_rooms.append((scene_name, room_id))
        
with open("/local-scratch/qiruiw/research/DeepPanoContext/data/s3d_metadata/scenes.txt", "w") as f:
    f.write(f"scene_name,room_id\n")
    for room in valid_rooms:
        f.write(f"{room[0]},{room[1]}\n")