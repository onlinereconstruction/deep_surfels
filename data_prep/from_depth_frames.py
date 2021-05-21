import argparse
import glob
import json
import os

import cc3d
import cv2
import cyfusion
import dt
import deep_surfel as dsurf
import numpy as np


def sdf_from_depth(destination_path, depth_maps, Ks, Rs, Ts, cam2world, resolution, box_size):
    truncation_factor = 10
    voxel_size = 1. / resolution
    truncation = truncation_factor * voxel_size

    # 1) generate SDF [-BS, BS]
    views = cyfusion.PyViews(depth_maps, Ks, Rs, Ts, Ks, cam2world)
    sdf = cyfusion.tsdf_cpu(views, resolution, resolution, resolution,
                            vx_size=voxel_size,
                            truncation=truncation,
                            unknown_is_free=False,
                            box_size=box_size)

    inside_inds = sdf <= 0
    sdf[inside_inds] = 1
    sdf[~inside_inds] = 0
    sdf = sdf.transpose(2, 1, 0)

    sdf = sdf.astype(np.int64)
    clean_occupancy_grid(sdf)
    sdf = sdf_from_occupancy_grid(sdf)

    translation = -box_size * (resolution - 1) / resolution
    scale = resolution / (2 * box_size)
    dsurf.save_sdf(destination_path, sdf, scale, translation)


def sdf_from_occupancy_grid(occupancy_grid):
    def compute_tsdf(grid):
        new_grid = np.copy(grid).astype(np.float64)

        new_grid[np.where(new_grid == 0.)] = 2.
        new_grid[np.where(new_grid == 1.)] = 0.
        new_grid[np.where(new_grid == 2.)] = 1.

        new_grid = 10.e6 * new_grid

        tsdf, i = dt.compute(new_grid)

        tsdf = np.sqrt(tsdf)

        return tsdf

    dist1 = compute_tsdf(occupancy_grid.astype(np.float64))
    dist1[dist1 > 0] -= 0.5
    dist2 = compute_tsdf(np.ones(dist1.shape) - occupancy_grid)
    dist2[dist2 > 0] -= 0.5
    sdf = np.copy(dist1 - dist2)

    return sdf


def clean_occupancy_grid(data):
    # clean occupancy grids from artifacts
    labels_out = cc3d.connected_components(data)  # 26-connected
    N = np.max(labels_out)
    max_label = 0
    max_label_count = 0
    for segid in range(1, N + 1):
        extracted_image = labels_out * (labels_out == segid)
        extracted_image[extracted_image != 0] = 1
        label_count = np.sum(extracted_image)
        if label_count > max_label_count:
            max_label = segid
            max_label_count = label_count
    data[labels_out != max_label] = 0.


# generate SDF
def generate_sdf(sdf_dst_path, files_root, geometry_resolution):
    def load_param(param_file):
        with open(param_file) as f:
            parameters = json.load(f)
            parameters['cam2world'] = dsurf.geometry.inv_extrinsics(np.array(parameters['RT']))[:3, :]
            return parameters

    def load_depth(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth.shape[-1] == 3 and len(depth.shape) == 3:
            depth = depth[..., 0]
        return depth

    files = glob.glob(os.path.join(files_root, '*'))
    frame_ids = list(set(map(lambda x: os.path.basename(x).split('_')[0], files)))

    depth_files = [os.path.join(files_root, f'{frame_id}_depth.exr') for frame_id in frame_ids]
    param_files = [os.path.join(files_root, f'{frame_id}_params.json') for frame_id in frame_ids]

    depth_maps = [load_depth(depth_file) for depth_file in depth_files]
    depth_maps = np.stack(depth_maps).astype(np.float32)

    params = [load_param(param_file) for param_file in param_files]
    Ks = np.stack([param['K'] for param in params]).astype(np.float32)
    Rs = np.stack([np.array(param['RT'])[:3, :3] for param in params]).astype(np.float32)
    Ts = np.stack([np.array(param['RT'])[:3, 3] for param in params]).astype(np.float32)
    cam2world = np.stack([param['cam2world'] for param in params]).astype(np.float32)

    # return depth_maps, Ks, Rs, Ts, cam2world
    box_size = 0.6
    sdf_from_depth(sdf_dst_path, depth_maps, Ks, Rs, Ts, cam2world, geometry_resolution, box_size)


def main():
    for category in os.listdir(DATASET_ROOT):
        for obj in os.listdir(os.path.join(DATASET_ROOT, category)):
            # shortened paths
            obj_root = os.path.join(DATASET_ROOT, category, obj)
            train_data_root = os.path.join(obj_root, 'train')
            sdf_path = os.path.join(obj_root, f'geometry_{SDF_GRID_RESOLUTION}.sdf')
            scene_path = os.path.join(obj_root, f'scene_{DEEPSURFEL_GRID_RESOLUTION}_{PATCH_RESOLUTION}.dsurf')

            # extract SDF
            generate_sdf(sdf_path, train_data_root, SDF_GRID_RESOLUTION)

            # create scene
            scene = dsurf.DeepSurfel.from_sdf(sdf_path, DEEPSURFEL_GRID_RESOLUTION, PATCH_RESOLUTION, CHANNELS)
            dsurf.export_mesh(scene_path.replace('.dsurf', '.ply'), scene)
            dsurf.save(scene_path, scene)
            print(f'Saved scene: {scene_path}')
            os.remove(sdf_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='./data_samples', help='Dataset root.')
    args = parser.parse_args()

    DATASET_ROOT = args.dataset_root

    DEEPSURFEL_GRID_RESOLUTION = 32
    PATCH_RESOLUTION = 2
    SDF_GRID_RESOLUTION = 32*2
    CHANNELS = 1
    main()
