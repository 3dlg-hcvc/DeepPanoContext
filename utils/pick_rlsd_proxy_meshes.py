import json
import trimesh
from math import inf

proxy_meshes = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/proxy_meshes.json"))

meshes = {}

for c in proxy_meshes:
    min_m = None
    min_v = inf
    for m in proxy_meshes[c]:
        mesh = trimesh.load(m, force='mesh', skip_materials=True)
        num_verts = len(mesh.vertices)
        if num_verts < min_v:
            min_m = m
            min_v = num_verts
        meshes[c] = min_m
        
import pdb; pdb.set_trace()
with open("/project/3dlg-hcvc/rlsd/data/annotations/proxy_meshes_min.json", "w") as f:
    json.dump(meshes, f, indent=4)