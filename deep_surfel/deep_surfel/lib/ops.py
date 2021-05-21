import torch

from .util import set_module
from ..deep_surfel import DeepSurfel


@set_module('deep_surfel')
def stack(deep_surfel_scenes):
    if len(deep_surfel_scenes) == 1:
        return deep_surfel_scenes[0]

    # checks for common parameters
    grid_resolution = deep_surfel_scenes[0].grid_resolution
    voxel_size = deep_surfel_scenes[0].voxel_size
    patch_resolution = deep_surfel_scenes[0].patch_resolution
    patch_size = deep_surfel_scenes[0].patch_size
    channels = deep_surfel_scenes[0].channels

    geometry_upsampling_factor = deep_surfel_scenes[0].geometry_upsampling_factor
    for ds in deep_surfel_scenes:
        assert ds.grid_resolution == grid_resolution
        assert ds.voxel_size == voxel_size
        assert ds.patch_resolution == patch_resolution
        assert ds.patch_size == patch_size
        assert ds.channels == channels
        assert ds.geometry_upsampling_factor == geometry_upsampling_factor

    ds = DeepSurfel(grid_resolution, voxel_size, patch_resolution[0], patch_size, channels)
    ds.geometry_upsampling_factor = geometry_upsampling_factor
    ds.features = torch.cat([x.features for x in deep_surfel_scenes], dim=0)
    ds.locations = torch.cat([x.locations for x in deep_surfel_scenes], dim=0)
    ds.orientations = torch.cat([x.orientations for x in deep_surfel_scenes], dim=0)
    ds.counts = torch.cat([x.counts for x in deep_surfel_scenes], dim=0)
    ds.normals = torch.cat([x.normals for x in deep_surfel_scenes], dim=0)

    ds.scale = torch.FloatTensor([x.scale for x in deep_surfel_scenes])
    ds.translate = torch.FloatTensor([x.translate for x in deep_surfel_scenes])

    ds.scene_id = [x.scene_id for x in deep_surfel_scenes]
    ds.batch_size = len(deep_surfel_scenes)

    return ds


@set_module('deep_surfel')
def split(scene):
    batch = scene.get_batch_size()
    if batch == 1:
        return [scene]

    def split_data(data, grid_resolution):
        feature_shape = tuple(data.shape[1:])
        grid_resolution = grid_resolution, grid_resolution, grid_resolution
        data = data.view(batch, *grid_resolution, *feature_shape)
        return [data[b].view(-1, *feature_shape).clone() for b in range(batch)]

    scenes = [
        DeepSurfel(scene.grid_resolution, scene.voxel_size, scene.patch_resolution[0], scene.patch_size, scene.channels)
        for _ in range(batch)
    ]

    features_list = split_data(scene.features, scene.grid_resolution)
    locations_list = split_data(scene.locations, scene.grid_resolution)
    orientations_list = split_data(scene.orientations, scene.grid_resolution)
    counts_list = split_data(scene.counts, scene.grid_resolution)
    normals_list = split_data(scene.normals, scene.grid_resolution * scene.geometry_upsampling_factor)

    for i in range(batch):
        scenes[i].geometry_upsampling_factor = scene.geometry_upsampling_factor

        scenes[i].features = features_list[i]
        scenes[i].locations = locations_list[i]
        scenes[i].orientations = orientations_list[i]
        scenes[i].counts = counts_list[i]
        scenes[i].normals = normals_list[i]

        scenes[i].scale = scene.scale[i].item()
        scenes[i].translate = scene.translate[i].item()

        scenes[i].scene_id = scene.scene_id[i]

    return scenes
