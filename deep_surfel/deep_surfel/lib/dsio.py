import pickle

import torch
import trimesh

from .util import set_module, create_quads


@set_module('deep_surfel')
def export_mesh(file, deep_surfel_scene, only_filled=False, features_as_colors=False, surfel_transformation=None):
    inside_inds = ~torch.isinf(deep_surfel_scene.locations).any(-1)
    if only_filled:
        inside_inds = inside_inds & (deep_surfel_scene.counts > 0)

    surfel_loc = deep_surfel_scene.locations[inside_inds]
    if features_as_colors:
        s_colors = deep_surfel_scene.features[inside_inds][..., :3]
        if surfel_transformation is not None:
            s_colors = surfel_transformation(s_colors)
    else:
        s_colors = torch.ones_like(surfel_loc) * 127

    surfel_orientations = deep_surfel_scene.orientations[inside_inds]

    s_vertices, s_faces = create_quads(surfel_loc, surfel_orientations, deep_surfel_scene.surfel_size)

    mesh = trimesh.Trimesh(
        vertices=s_vertices.cpu().numpy(),
        faces=s_faces.cpu().numpy(),
        vertex_normals=s_vertices.repeat_interleave(4, dim=0).cpu().numpy(),
        vertex_colors=s_colors.repeat_interleave(4, dim=0).cpu().numpy()
    )
    mesh.export(file)


@set_module('deep_surfel')
def save(file, scene):
    if not file.endswith('.dsurf'):
        file = f'{file}.dsurf'

    with open(file, 'wb') as f:
        pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)


@set_module('deep_surfel')
def load(file):
    if not file.endswith('.dsurf'):
        file = f'{file}.dsurf'

    with open(file, 'rb') as f:
        scene = pickle.load(f)
    return scene


@set_module('deep_surfel')
def save_sdf(dst_file, sdf, scale, translation):
    if not dst_file.endswith('.sdf'):
        dst_file = f'{dst_file}.sdf'

    with open(dst_file, 'wb') as f:
        pickle.dump((sdf, scale, translation), f, protocol=pickle.HIGHEST_PROTOCOL)


@set_module('deep_surfel')
def load_sdf(src_file):
    if not src_file.endswith('.sdf'):
        src_file = f'{src_file}.sdf'

    with open(src_file, 'rb') as f:
        sdf, scale, translation = pickle.load(f)

    return sdf, scale, translation
