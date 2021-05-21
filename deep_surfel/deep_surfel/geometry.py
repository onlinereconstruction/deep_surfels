import torch
import numpy as np

from torch.nn import functional as F


def create_uv(width, height):
    uv = np.flip(np.mgrid[height[0]:height[1], width[0]:width[1]].astype(np.int32), axis=0).copy()
    return uv.reshape((2, -1)).T


def create_perpendicular_vectors_vectorized(normals):  # Nx3 tensor
    def handle_zeros(n_vec):
        row_inds = torch.arange(n_vec.shape[0], device=normals.device, dtype=torch.long)
        max_inds = torch.abs(n_vec).argmax(dim=-1, keepdim=True)
        zero_inds = torch.arange(3, device=normals.device, dtype=torch.long).view(1, 3).repeat(n_vec.shape[0], 1)
        zero_inds = zero_inds[torch.where(zero_inds != max_inds)].view(n_vec.shape[0], -1)
        vec_x, vec_y = torch.zeros_like(n_vec), torch.zeros_like(n_vec)
        vec_x[row_inds, zero_inds[:, 0]] = 1
        vec_y[row_inds, zero_inds[:, 1]] = 1
        return vec_x, vec_y

    def handle_nonzeros(n_vec):
        row_inds = torch.arange(n_vec.shape[0], device=normals.device, dtype=torch.long)
        vec = torch.zeros_like(n_vec)
        max_ind = torch.abs(n_vec).argmax(dim=-1)
        vec[row_inds, max_ind] = n_vec[row_inds, max_ind]
        vec_x = torch.cross(vec, n_vec, dim=-1)
        vec_y = torch.cross(vec_x, n_vec, dim=-1)

        vec_y = F.normalize(vec_y, dim=-1)
        vec_x = F.normalize(vec_x, dim=-1)
        return vec_x, vec_y

    vec_x = torch.empty_like(normals)
    vec_y = torch.empty_like(normals)
    zero_inds = (normals == 0).sum(axis=-1) == 2
    non_zero_inds = ~zero_inds

    if zero_inds.any():
        vec_x[zero_inds], vec_y[zero_inds] = handle_zeros(normals[zero_inds])
    if non_zero_inds.any():
        vec_x[non_zero_inds], vec_y[non_zero_inds] = handle_nonzeros(normals[non_zero_inds])

    return vec_x, vec_y


def plane_points_to_3d_vectorized(normal_vec, local_coords, dhw):
    assert normal_vec.shape[0] == local_coords.shape[0] == dhw.shape[0]
    assert normal_vec.shape[1] == dhw.shape[1] == 3 and local_coords.shape[1] == 2

    vec_d, vec_w = create_perpendicular_vectors_vectorized(normal_vec)
    T_inv = torch.cat((vec_d, normal_vec, vec_w), dim=-1).view(-1, 3, 3)  # vectors are rows

    points2D = torch.cat((
        local_coords,
        torch.zeros((local_coords.shape[0], 1), dtype=local_coords.dtype, device=local_coords.device)), dim=1)
    points2D = torch.stack((points2D[:, 1], points2D[:, 2], points2D[:, 0])).t()  # xyz -> dhw space
    points3D = torch.matmul(points2D.unsqueeze(1), T_inv).squeeze() + dhw
    return points3D


def trilinear_interpolation(points, grid):
    dhw_inds, interpolation_weights = add_trilinear_neigh_points(points)
    interpolated_values = \
        grid[dhw_inds[..., 0].reshape(-1),
             dhw_inds[..., 1].reshape(-1),
             dhw_inds[..., 2].reshape(-1)] * interpolation_weights.view(-1, 1)
    interpolated_values = interpolated_values.view(-1, 8, grid.shape[-1]).sum(1)
    return interpolated_values


def add_trilinear_neigh_points(xyz):
    """ Add neighbouring points. The first point is central

    Args:
         xyz (torch.Tensor): query points in the grid space (-1, 3)

    Returns:
        dhw inds (np.ndarray): grid points (-1, 8, 3)
        dhw weights (np.ndarray): grid points (-1, 8, 1)
    """
    # new code
    points = xyz
    # get indices
    indices = torch.floor(points)

    # compute interpolation distance
    df = torch.abs(points - indices)

    # get interpolation indices
    xx, yy, zz = torch.meshgrid([torch.arange(0, 2), torch.arange(0, 2), torch.arange(0, 2)])

    xx = xx.contiguous().view(8)
    yy = yy.contiguous().view(8)
    zz = zz.contiguous().view(8)

    shift = torch.stack([xx, yy, zz], dim=1)

    shift = shift.to(points.device)

    # reshape
    shift = shift.unsqueeze_(0)
    indices = indices.unsqueeze_(1)

    # compute indices
    indices = indices + shift

    # init weights
    weights = torch.zeros_like(indices).sum(dim=-1)

    # compute weights
    weights[:, 0] = (1 - df[:, 0]) * (1 - df[:, 1]) * (1 - df[:, 2])
    weights[:, 1] = (1 - df[:, 0]) * (1 - df[:, 1]) * df[:, 2]
    weights[:, 2] = (1 - df[:, 0]) * df[:, 1] * (1 - df[:, 2])
    weights[:, 3] = (1 - df[:, 0]) * df[:, 1] * df[:, 2]
    weights[:, 4] = df[:, 0] * (1 - df[:, 1]) * (1 - df[:, 2])
    weights[:, 5] = df[:, 0] * (1 - df[:, 1]) * df[:, 2]
    weights[:, 6] = df[:, 0] * df[:, 1] * (1 - df[:, 2])
    weights[:, 7] = df[:, 0] * df[:, 1] * df[:, 2]

    weights = weights.unsqueeze_(-1)

    return indices.view(-1, 8, 3).long(), weights.float().view(-1, 8, 1)


def find_iso_surface(points, directions, sdf_grid, geometry_upsampling_factor=1,
                     normal_marching=False):  # S3 points are in [0, 1] space
    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    if len(sdf_grid.shape) == 3:
        sdf_grid = sdf_grid.unsqueeze(-1)

    max_step = 4  # / max(sdf_grid.shape)
    original_points = points.clone()
    while True:
        sdfs = trilinear_interpolation(points * geometry_upsampling_factor, sdf_grid).view(-1)
        inds = (torch.abs(sdfs) > 1e-5) & (torch.abs(points - original_points) <= max_step).all(-1)
        mask = mask | ~inds
        if mask.all():
            break

        points[inds] += sdfs[inds].view(-1, 1) * directions[inds]
        if not normal_marching:
            return points

    return points


def get_patch_coordinates(patch_resolution, patch_size):
    puv = create_uv(width=(0, patch_resolution), height=(0, patch_resolution))
    puv_coords = ((puv + 0.5) / patch_resolution - 0.5) * patch_size
    return torch.from_numpy(puv_coords).to(dtype=torch.float32)


def shift_points(points, sdf_grid, normal_grid, geometry_upsampling_factor=1, prev_directions=None):
    directions = -trilinear_interpolation(points * geometry_upsampling_factor, normal_grid)
    directions = F.normalize(directions, dim=-1)
    if prev_directions is not None:
        inds_to_overwrite = torch.where((directions == 0).all(-1))
        directions[inds_to_overwrite] = prev_directions[inds_to_overwrite]
    points = find_iso_surface(points, directions, sdf_grid, geometry_upsampling_factor)
    return points, directions


def surfel_locations(dhw_inds, sdf_grid, normal_grid, voxel_size,
                     patch_resolution, patch_size, geometry_upsampling_factor):  # [0, 1] space
    n_shifts = num_divisible_by_2(patch_resolution)
    patch_resolution_list = [2 for _ in range(n_shifts)]
    if int(patch_resolution / 2 ** n_shifts) != 1:
        patch_resolution_list = patch_resolution_list + [int(patch_resolution / 2 ** n_shifts)]
    patch_size_list = [patch_size / (2 ** i) for i in range(len(patch_resolution_list))]

    d_inds, h_inds, w_inds = dhw_inds[0].long(), dhw_inds[1].long(), dhw_inds[2].long()
    points = voxel_size * torch.stack((d_inds, h_inds, w_inds)).t().float()  # S3

    voxel_centers = points.detach().clone()
    points, dirs = shift_points(points, sdf_grid, normal_grid, geometry_upsampling_factor)
    for i in range(len(patch_resolution_list)):
        # subdivide patch
        puv_coords = get_patch_coordinates(patch_resolution_list[i], patch_size_list[i]).repeat(points.shape[0], 1)
        points = points.repeat_interleave(patch_resolution_list[i] ** 2, dim=0).view(-1, 3)  # (S4)3
        dirs = dirs.repeat_interleave(patch_resolution_list[i] ** 2, dim=0).view(-1, 3)  # (S4)3
        points = plane_points_to_3d_vectorized(dirs, puv_coords, points)  # (ST)3

        points, dirs = shift_points(points, sdf_grid, normal_grid, geometry_upsampling_factor, dirs)

    # remove points that are outside their voxel cells
    voxel_centers = voxel_centers.repeat_interleave(patch_resolution ** 2, dim=0).view(-1, 3)
    shifted_points = torch.abs(voxel_centers - points).max(dim=-1)[0]
    outside_voxel_inds = shifted_points > voxel_size * 1
    directions = -dirs
    directions[outside_voxel_inds] = 0
    points[outside_voxel_inds] = float('inf')

    return points, directions


def surfel_default_locations(dhw_inds, voxel_size, patch_resolution, patch_size):
    d_inds, h_inds, w_inds = dhw_inds[0].long(), dhw_inds[1].long(), dhw_inds[2].long()
    points = voxel_size * torch.stack((d_inds, h_inds, w_inds)).t().float()  # S3

    puv_coords = get_patch_coordinates(patch_resolution, patch_size).repeat(points.shape[0], 1)
    points = points.repeat_interleave(patch_resolution ** 2, dim=0).view(-1, 3)  # (S4)3
    dirs = torch.ones_like(points)  # (S4)3
    dirs = F.normalize(dirs, dim=-1)
    points = plane_points_to_3d_vectorized(dirs, puv_coords, points)  # (ST)3
    return points, dirs


def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1

    return i


def inv_extrinsics(extrinsics):
    assert type(extrinsics) == np.ndarray or torch.is_tensor(extrinsics)

    if torch.is_tensor(extrinsics):
        cam2world = torch.eye(4)
        cam2world[:3, :3] = extrinsics[:3, :3].t()
        cam2world[:3, 3] = torch.matmul(extrinsics[:3, :3].t(), -extrinsics[:3, 3]).reshape(-1)
    else:
        cam2world = np.eye(4)
        cam2world[:3, :3] = extrinsics[:3, :3].T
        cam2world[:3, 3] = np.matmul(extrinsics[:3, :3].T, -extrinsics[:3, 3]).reshape(-1)

    return cam2world


def inv_cam2world(cam2world):
    assert type(cam2world) == np.ndarray or torch.is_tensor(cam2world)

    if torch.is_tensor(cam2world):
        extrinsics = torch.eye(4)
        extrinsics[:3, :3] = cam2world[:3, :3].t()
        extrinsics[:3, 3] = torch.matmul(extrinsics[:3, :3], -cam2world[:3, 3]).reshape(-1)
    else:  # numpy
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = cam2world[:3, :3].T
        extrinsics[:3, 3] = np.matmul(extrinsics[:3, :3], -cam2world[:3, 3]).reshape(-1)

    return extrinsics


def sdf2normal_grid(sdf):
    normal_grid = np.zeros((*sdf.shape, 3), dtype=np.float32)

    d_diff = sdf[2:, :, :] - sdf[:-2, :, :]
    h_diff = sdf[:, 2:, :] - sdf[:, :-2, :]
    w_diff = sdf[:, :, 2:] - sdf[:, :, :-2]

    normal_grid[1:-1, :, :, 0] = d_diff
    normal_grid[:, 1:-1, :, 1] = h_diff
    normal_grid[:, :, 1:-1, 2] = w_diff

    norm = np.linalg.norm(normal_grid, axis=-1)
    inds = norm != 0
    normal_grid[inds] = normal_grid[inds] / norm[inds, None]
    return normal_grid
