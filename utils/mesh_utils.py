import os
import re

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

import torch
from mesh_intersection.bvh_search_tree import BVH


def save_mesh(mesh, save_path):
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        save_path = os.path.join(os.path.dirname(save_path),
                                 os.path.basename(os.path.splitext(save_path)[0]),
                                 os.path.basename(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trimesh.exchange.export.export_mesh(mesh, save_path)


def load_mesh(path, mesh_only=False):
    # mesh = trimesh.load_mesh(path)
    # if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.load(path, force='mesh', skip_materials=True)
    if mesh_only:
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return mesh


class MeshExtractor:
    def extract_mesh(self, *args, **kwargs):
        raise NotImplementedError


class MeshIO(dict):
    def __init__(self, meshes=None):
        if meshes is None:
            meshes = {}
        self.mesh_path = {}
        super().__init__(meshes)

    @classmethod
    def from_file(cls, key_path_pair: (dict, list)):
        mesh_io = cls()
        if isinstance(key_path_pair, list):
            key_path_pair = {i: p for i, p in enumerate(key_path_pair)}
        mesh_io.mesh_path = key_path_pair
        return mesh_io

    def __getitem__(self, item):
        if item not in super().keys():
            mesh = load_mesh(self.mesh_path[item], mesh_only=True)
            super().__setitem__(item, mesh)
        return super().__getitem__(item)

    def load(self):
        for k in self.mesh_path.keys():
            self.__getitem__(k)
        return self

    def merge(self):
        return sum([m for m in self.values()]) if self else trimesh.Trimesh()

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        for k, v in self.items():
            save_mesh(v, os.path.join(folder, f"{k}.obj"))


#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)


# determinant of matrix a
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]


# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0.:
        return (0., 0., 0.)
    else:
        return (x/magnitude, y/magnitude, z/magnitude)


#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


#area of polygon poly
def get_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]

    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def calculate_face_area(data):

    face_areas = []

    for face in data['f']:
        vid_in_face = [int(item.split('/')[0]) for item in face]
        face_area = get_area(data['v'][np.array(vid_in_face) - 1,:3].tolist())
        face_areas.append(face_area)

    return face_areas


def sample_pnts_from_obj(data, n_pnts = 5000, mode = 'uniform'):
    # sample points on each object mesh.

    flags = data.keys()

    all_pnts = data['v'][:,:3]

    area_list = np.array(calculate_face_area(data))
    distribution = area_list/np.sum(area_list)

    # sample points the probability depends on the face area
    new_pnts = []
    if mode == 'random':

        random_face_ids = np.random.choice(len(data['f']), n_pnts, replace=True, p=distribution)
        random_face_ids, sample_counts = np.unique(random_face_ids, return_counts=True)

        for face_id, sample_count in zip(random_face_ids, sample_counts):

            face = data['f'][face_id]

            vid_in_face = [int(item.split('/')[0]) for item in face]

            weights = np.diff(np.sort(np.vstack(
                [np.zeros((1, sample_count)), np.random.uniform(0, 1, size=(len(vid_in_face) - 1, sample_count)),
                 np.ones((1, sample_count))]), axis=0), axis=0)

            new_pnt = all_pnts[np.array(vid_in_face) - 1].T.dot(weights)

            if 'vn' in flags:
                nid_in_face = [int(item.split('/')[2]) for item in face]
                new_normal = data['vn'][np.array(nid_in_face)-1].T.dot(weights)
                new_pnt = np.hstack([new_pnt, new_normal])


            new_pnts.append(new_pnt.T)

        random_pnts = np.vstack(new_pnts)

    else:

        for face_idx, face in enumerate(data['f']):
            vid_in_face = [int(item.split('/')[0]) for item in face]

            n_pnts_on_face = distribution[face_idx] * n_pnts

            if n_pnts_on_face < 1:
                continue

            dim = len(vid_in_face)
            npnts_dim = (np.math.factorial(dim - 1)*n_pnts_on_face)**(1/(dim-1))
            npnts_dim = int(npnts_dim)

            weights = np.stack(np.meshgrid(*[np.linspace(0, 1, npnts_dim) for _ in range(dim - 1)]), 0)
            weights = weights.reshape(dim - 1, -1)
            last_column = 1 - weights.sum(0)
            weights = np.vstack([weights, last_column])
            weights = weights[:, last_column >= 0]

            new_pnt = (all_pnts[np.array(vid_in_face) - 1].T.dot(weights)).T

            if 'vn' in flags:
                nid_in_face = [int(item.split('/')[2]) for item in face]
                new_normal = data['vn'][np.array(nid_in_face) - 1].T.dot(weights)
                new_pnt = np.hstack([new_pnt, new_normal])

            new_pnts.append(new_pnt)

        random_pnts = np.vstack(new_pnts)

    return random_pnts


def normalize_to_unit_square(points, keep_ratio=True):
    centre = (points.max(0) + points.min(0)) / 2.
    point_shapenet = points - centre

    if keep_ratio:
        scale = point_shapenet.max()
    else:
        scale = point_shapenet.max(0)
    point_shapenet = point_shapenet / scale

    return point_shapenet, centre, scale


def read_obj(model_path, flags=('v')):
    fid = open(model_path, 'r')

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        # line = line.strip().split(' ')
        line = re.split('\s+', line.strip())
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float)

    return data


def write_obj(objfile, data):
    with open(objfile, 'w+') as file:
        for item in data['v']:
            file.write('v' + ' %f' * len(item) % tuple(item) + '\n')

        for item in data['f']:
            file.write('f' + ' %s' * len(item) % tuple(item) + '\n')


def create_layout_mesh(data, color=None, radius=0.025, texture=True):
    from .layout_utils import layout_line_segment_indexes
    from .igibson_utils import IGScene
    if 'layout' not in data or (
            'manhattan_world' not in data['layout']
            and 'total3d' not in data['layout']
    ):
        return None
    if 'total3d' in data['layout']:
        mesh = create_layout_mesh(data['layout']['total3d'], color=color)
    elif 'manhattan_world' in data['layout']:
        mesh = []
        layout_points = data['layout']['manhattan_world']
        layout_lines = layout_line_segment_indexes(len(layout_points) // 2)
        for indexes in layout_lines:
            line = layout_points[indexes]
            line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
            mesh.append(line_mesh)
        mesh = sum(mesh)
        if color is not None:
            mesh = IGScene.colorize_mesh_for_igibson(mesh, color, texture)
    return mesh


def create_bdb3d_mesh(bdb3d, color=None, radius=0.05, texture=True):
    from .igibson_utils import IGScene
    from .transform_utils import bdb3d_corners
    corners = bdb3d_corners(bdb3d)
    corners_box = corners.reshape(2, 2, 2, 3)
    mesh = []
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                line = corners_box[idx1], corners_box[idx2]
                line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
                mesh.append(line_mesh)
    # for idx1, idx2 in [(0, 5), (1, 4)]:
    #     line = corners[idx1], corners[idx2]
    #     line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
    #     mesh.append(line_mesh)
    mesh = sum(mesh)
    if color is not None:
        mesh = IGScene.colorize_mesh_for_igibson(mesh, color, texture)
    return mesh


def write_ply_rgb_face(points, colors, faces, filename, text=True):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    faces = [((faces[i,0], faces[i,1], faces[i,2]),) for i in range(faces.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    face = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(face, 'face', comments=['faces'])
    PlyData([ele1, ele2], text=text).write(filename)
    
    
def mesh_collision(mesh1, mesh2):
    num_mesh1_faces = len(mesh1.faces)
    mesh = trimesh.util.concatenate([mesh1, mesh2])
    vertices = torch.tensor(mesh.vertices,
                            dtype=torch.float32).to('cuda')
    faces = torch.tensor(mesh.faces.astype(np.int64),
                         dtype=torch.long).to('cuda')
    
    triangles = vertices[faces].unsqueeze(dim=0)

    m = BVH(max_collisions=16)

    torch.cuda.synchronize()
    outputs = m(triangles)
    torch.cuda.synchronize()

    outputs = outputs.detach().cpu().numpy().squeeze()
    collisions = outputs[outputs[:, 0] >= 0, :]
    
    face_diffs = np.sign(collisions - num_mesh1_faces + 0.5)
    valid = (face_diffs[:, 0] * face_diffs[:, 1]) < 0
    collisions = collisions[valid]
    
    # all_collisions = np.unique(collisions.reshape(-1))
    # mesh1_collisions = all_collisions[all_collisions < num_mesh1_faces]
    # mesh2_collisions = all_collisions[all_collisions >= num_mesh1_faces]
    
    return len(collisions) > 0