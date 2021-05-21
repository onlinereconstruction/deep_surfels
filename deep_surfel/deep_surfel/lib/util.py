import os

import numpy as np
import torch

from ..geometry import plane_points_to_3d_vectorized


def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func

    return decorator


def cond_remove(file):
    if os.path.exists(file):
        os.remove(file)


def create_quads(location, normal, plane_size):
    p2 = plane_size / 2.
    plane = torch.from_numpy(np.array([
        [p2, -p2],
        [p2, p2],
        [-p2, p2],
        [-p2, -p2]
    ])).to(device=location.device, dtype=location.dtype)
    n_vertices = 4
    t_dirs = normal.repeat_interleave(n_vertices, dim=0)  # (NV)3
    t_centers = location.repeat_interleave(n_vertices, dim=0)  # (NV)3
    plane = plane.repeat(location.shape[0], 1)  # (NV)3
    t_vertices = plane_points_to_3d_vectorized(t_dirs, plane, t_centers)  # (NV)3

    inds = n_vertices * torch.arange(location.shape[0]).unsqueeze(-1).repeat_interleave(n_vertices, dim=-1)
    inds = inds + torch.arange(n_vertices).view(1, -1)
    t_faces = inds.view(-1, n_vertices)
    return t_vertices, t_faces  # (NV)3,  N3,  NV


def rescale_mesh(vertices, grid_resolution):
    translation = vertices.min()
    vertices = vertices - translation

    mesh_size = vertices.max()
    scale = (grid_resolution - 1) / (mesh_size + 0.2 * mesh_size)
    vertices *= scale

    return translation, scale
