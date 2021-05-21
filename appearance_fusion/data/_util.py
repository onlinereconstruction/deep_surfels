import json
from glob import glob
from os.path import join, basename
from random import shuffle

import numpy as np
import torch
from torch.nn import functional as F
import deep_surfel as dsurf


def get_frame_ids(scene_root, sort=False):
    files = glob(join(scene_root, '*_img.png'))
    files = list(map(lambda x: basename(x).split('_')[0], files))
    if sort:
        files = sorted(files)
    else:
        shuffle(files)
    return files


def find_closest_frames(dir_path, frame_ids, dont_toggle_yz):
    def load_extrinsics(frame_id):
        param_path = join(dir_path, f'{frame_id}_params.json')
        _, _, cam2world = load_params(param_path, dont_toggle_yz)
        cam_position = cam2world[:3, 3].cpu().numpy()

        return cam_position

    cam_positions = {frame_id: load_extrinsics(frame_id) for frame_id in frame_ids}
    closest_frame_ids = []
    for frame_id in frame_ids:
        min_distance, closest_frame_id = 1e9, frame_id
        for iter_frame_id in frame_ids:
            if iter_frame_id == frame_id:
                continue

            new_dist = np.linalg.norm(cam_positions[frame_id] - cam_positions[iter_frame_id])
            if new_dist < min_distance or closest_frame_id is None:
                min_distance = new_dist
                closest_frame_id = iter_frame_id

        closest_frame_ids.append(closest_frame_id)

    return closest_frame_ids


def insert_batch_dimension(points, batch_value=0):
    if batch_value == 0:
        batch_inds = torch.zeros((points.shape[1], 1), dtype=points.dtype, device=points.device)
    else:
        batch_inds = torch.full((points.shape[1], 1), fill_value=batch_value, dtype=points.dtype, device=points.device)

    points = points.squeeze(0)
    points = torch.cat((batch_inds, points), dim=1)
    return points


def prepare_frame_single_batch(frame, device=None):
    frame['points'] = insert_batch_dimension(frame['points'])
    frame['ray_dirs'] = frame['ray_dirs'].squeeze(0)
    frame['image'] = frame['image'].squeeze(0)
    frame['pixel_size'] = frame['pixel_size'].squeeze(0)
    frame['depth'] = frame['depth'].squeeze(0)

    if device is not None:
        frame2device(frame, device)
    return frame


def frame2device(frame, device):
    for key in frame.keys():
        if key != 'frame_id':
            frame[key] = frame[key].to(device=device)


def stack_frames(frames):
    frame = {'frame_id': [f['frame_id'] for f in frames]}
    for key in ['gt_image', 'mask', 'points_mask']:
        frame[key] = torch.cat([f[key] for f in frames], dim=0)

    for key in ['depth', 'image', 'ray_dirs', 'pixel_size']:
        frame[key] = torch.cat([f[key].squeeze(0) for f in frames], dim=0)

    frame['points'] = torch.cat([insert_batch_dimension(f['points'], i) for i, f in enumerate(frames)], dim=0)

    return frame


def load_params(params_file, dont_toggle_yz=False):
    with open(params_file) as f:
        parameters = json.load(f)
    K = np.array(parameters['K'])
    if 'camera2world' in parameters:
        cam2world = np.array(parameters['camera2world'])
        extrinsics = np.linalg.inv(cam2world)
    else:
        K, extrinsics = np.array(parameters['K']), np.array(parameters['RT'])

        cam2world = dsurf.geometry.inv_extrinsics(extrinsics)
        # flip axis (when frames are rendered by blender)
        if not dont_toggle_yz:
            mToggle_YZ = np.array((  # toggle xz and flip z; X' = X, Y' = Z, Z' = -Y
                (1, 0, 0, 0),
                (0, 0, 1, 0),
                (0, -1, 0, 0),
                (0, 0, 0, 1),
            ))
            cam2world = np.matmul(mToggle_YZ, cam2world)
        extrinsics = np.linalg.inv(cam2world)

    K, extrinsics, cam2world = torch.from_numpy(K), torch.from_numpy(extrinsics), torch.from_numpy(cam2world)
    return K, extrinsics, cam2world


def get_scene_ids(data_root):
    scene_ids = []
    cat_directories = glob(join(data_root, '*'))
    for directory in cat_directories:
        category = basename(directory)
        scene_dirs = glob(join(directory, '*'))
        for scene_dir in scene_dirs:
            scene_ids.append(join(category, basename(scene_dir)))

    return scene_ids


def pix2coord(uvd, K):  # verified
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if type(uvd) == np.ndarray:
        camera_coord = np.ones((4, uvd.shape[0]))
    elif torch.is_tensor(uvd):
        camera_coord = torch.ones((4, uvd.shape[0]), dtype=uvd.dtype, device=uvd.device)
    else:
        raise Exception('Unsupported data type.')

    camera_coord[0] = uvd[:, 2] * (uvd[:, 0] - cx) / fx
    camera_coord[1] = uvd[:, 2] * (uvd[:, 1] - cy) / fy
    camera_coord[2] = uvd[:, 2]
    camera_coord[:, (uvd[:, 2] == np.inf)] = np.inf

    return camera_coord


def uvd2xyz(uvd, cam2world, K):
    """Maps image coordinates (uvd) to voxel coordinates.

    Args:
        uvd (torch.Tensor): image coordinates of shape (number_of_samples, 3)
        cam2world (torch.Tensor): camera matrix of shape (4, 4)
        K (torch.Tensor): camera intrinsics of shape (3, 3)
    Returns:
        XYZ
    """
    xyzh = pix2coord(uvd, K)
    XYZ = torch.matmul(cam2world, xyzh)[:3, :]
    XYZ[:, (uvd[:, 2] == np.inf)] = np.inf
    return XYZ.t()


def get_ray_dirs(uvd, cam2world, K):
    uv_d1 = torch.ones_like(uvd)
    uv_d1[:, :2] = uvd[:, :2]
    rays = uvd2xyz(uvd, cam2world, K)
    rays[torch.isnan(rays)] = 1
    rays = F.normalize(rays, dim=-1)
    return rays
