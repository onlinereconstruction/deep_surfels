import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import deep_surfel as dsurf
from .geometry import surfel_locations, add_trilinear_neigh_points, surfel_default_locations, sdf2normal_grid


class DeepSurfel:
    def __init__(self, grid_resolution, voxel_size, patch_resolution, patch_size, channels):
        self.depth, self.height, self.width = grid_resolution, grid_resolution, grid_resolution
        self.grid_resolution = grid_resolution
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.patch_samples = patch_resolution ** 2
        self.surfel_size = patch_size / patch_resolution

        self.patch_resolution = patch_resolution, patch_resolution
        self.n_features = patch_resolution ** 2
        self.channels = channels
        self.geometry_upsampling_factor = 1

        self.features = None
        self.locations = None
        self.orientations = None
        self.counts = None
        self.normals = None

        self._scale = None
        self._translate = None

        self.batch_size = 1
        self.scene_id = None

    @property
    def scale(self):
        return self._scale

    @property
    def translate(self):
        return self._translate

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @translate.setter
    def translate(self, translate):
        self._translate = translate

    @classmethod
    def from_sdf(cls, sdf_path, resolution, patch_resolution, channels, patch_size=1., texels_at_infinity=True):
        sdf, scale, translation = dsurf.load_sdf(sdf_path)
        assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2] and len(sdf.shape) == 3
        assert sdf.shape[0] >= resolution and sdf.shape[0] % resolution == 0

        geometry_upsampling_factor = int(sdf.shape[0] / resolution)
        sdf = sdf / geometry_upsampling_factor
        scale = scale / geometry_upsampling_factor

        normal_grid = torch.from_numpy(sdf2normal_grid(sdf))
        sdf = torch.from_numpy(sdf)  # / resolution

        occupied_voxels = torch.abs(sdf) <= 5
        if geometry_upsampling_factor:
            occupied_voxels = occupied_voxels.view(1, 1, *sdf.shape).float()
            maxpool = nn.MaxPool3d(kernel_size=geometry_upsampling_factor, stride=geometry_upsampling_factor, padding=0)
            occupied_voxels = maxpool(occupied_voxels).squeeze(0).squeeze(0).bool()

        # instantiate DeepSurfel
        ds = DeepSurfel(resolution, 1, patch_resolution, patch_size, channels)

        dhw_inds = torch.where(occupied_voxels)
        s_surfel_loc, s_dirs = \
            surfel_locations(dhw_inds, sdf, normal_grid, 1, patch_resolution, patch_size, geometry_upsampling_factor)

        grid_elements = resolution ** 3

        ds.locations = torch.ones((grid_elements, ds.patch_samples, 3)) * float('inf')
        ds.orientations = torch.zeros((grid_elements, ds.patch_samples, 3))

        inds = ds._bdhw2idx(torch.stack(dhw_inds).t())
        ds.locations[inds] = s_surfel_loc.view(-1, ds.patch_samples, 3)
        ds.orientations[inds] = s_dirs.view(-1, ds.patch_samples, 3)

        ds.normals = normal_grid.view((resolution * geometry_upsampling_factor) ** 3, 3)
        ds.counts = torch.zeros((grid_elements, ds.patch_samples))
        ds.features = torch.zeros((grid_elements, ds.patch_samples, ds.channels))
        ds.geometry_upsampling_factor = geometry_upsampling_factor

        ds.translate = translation
        ds.scale = scale

        if not texels_at_infinity:
            infty_voxels = torch.any(torch.isinf(ds.locations.view(grid_elements, -1)), dim=-1)
            dhw_inds = torch.where(infty_voxels.view(resolution, resolution, resolution))
            s_surfel_loc, s_dirs = surfel_default_locations(dhw_inds, 1, patch_resolution, patch_size)

            ds.locations[infty_voxels] = s_surfel_loc.view(-1, ds.patch_samples, 3)
            ds.orientations[infty_voxels] = s_dirs.view(-1, ds.patch_samples, 3)

        return ds

    def reset(self, new_n_channels=None):
        self.counts = torch.zeros_like(self.counts)

        if new_n_channels is None:
            self.features = torch.zeros_like(self.features)
        else:
            self.channels = new_n_channels
            shape = *self.features.shape[:-1], new_n_channels
            self.features = torch.zeros(shape, dtype=self.features.dtype, device=self.features.device)

    def _wc2lc(self, world_coords, inplace=False):
        assert world_coords.shape[1] in [3, 4]

        if world_coords.shape[1] == 3:  # add batch indices
            batch_inds = torch.zeros(world_coords.shape[0], device=world_coords.device, dtype=world_coords.dtype)
            world_coords = torch.cat((batch_inds.view(-1, 1), world_coords), dim=1)

        world_coords = self.world2local_transform(world_coords, inplace)
        return world_coords

    def world2local_transform(self, world_coords, inplace=False):
        if not inplace:
            world_coords = world_coords.clone()

        if self.batch_size == 1:
            world_coords[:, 1:] = (world_coords[:, 1:] - self.translate) * self.scale
        else:
            self._translate = self._translate.to(device=world_coords.device)
            self._scale = self._scale.to(device=world_coords.device)

            b = world_coords[:, 0].long()
            world_coords[:, 1:] = (world_coords[:, 1:] - self.translate[b].view(-1, 1)) * self.scale[b].view(-1, 1)

        return world_coords

    def _lc2wc(self, local_coords, inplace=False):
        assert local_coords.shape[1] == 4

        if not inplace:
            local_coords = local_coords.clone()

        if self.batch_size == 1:
            local_coords[:, 1:] = local_coords[:, 1:] / self.scale + self.translate
        else:
            self._translate.to(device=local_coords.device)
            self._scale.to(device=local_coords.device)

            b = local_coords[:, 0].long()
            local_coords[:, 1:] = local_coords[:, 1:] / self.scale[b].view(-1, 1) + self.translate[b].view(-1, 1)

        return local_coords

    def _bdhw2idx(self, bdhw, resolution=None):
        if resolution is None:
            resolution = self.grid_resolution

        bdhw = bdhw.clamp(0, resolution - 1)
        if bdhw.shape[1] == 4:
            inds = ((bdhw[:, 0] * resolution + bdhw[:, 1]) * resolution + bdhw[:, 2]) * resolution + bdhw[:, 3]
        elif bdhw.shape[1] == 3:
            inds = (bdhw[:, 0] * resolution + bdhw[:, 1]) * resolution + bdhw[:, 2]
        else:
            raise Exception('Invalid indexes.')
        return inds.long()

    def _idx2bdhw(self, inds):
        w = inds % self.width
        inds = (inds - w) / self.width

        h = inds % self.height
        inds = (inds - h) / self.height

        d = inds % self.depth
        inds = (inds - d) / self.depth

        b = inds

        return torch.stack((b, d, h, w)).t()

    def get_batch_size(self):
        # return self.counts.view(-1, self.depth, self.height, self.width).shape[0]
        return self.batch_size

    def _projection(self, points, box_size, n_layers, mode='read', return_read_weights=False):
        assert mode in ['read', 'write', 'fuse']
        device = points.device

        points = self._wc2lc(points)
        if self.batch_size == 1:
            box_size = box_size * self.scale
        else:
            self._scale.to(device=points.device)
            box_size = box_size * self.scale[points[:, 0].long()].view(-1, 1)

        layers = n_layers * 2 + 1

        # estimate points
        normals = self._estimate_surface_direction(points)  # (BS)3
        points = self._estimate_layer_locations(points, normals, n_layers)  # (BSL)4
        inds = self._bdhw2idx((points + 0.5).int())  # (BSL)4

        points = points[:, 1:].repeat_interleave(self.patch_samples, dim=0)  # (BSL)4 -> (BSL)3 ->   # (BSLPP)3

        # compute weights
        locations = self.locations[inds].view(-1, 3).to(device=device)  # (BSLPP)3
        distances = torch.abs(points - locations).max(dim=-1)[0]  # (BSLPP)
        distances = distances.view(-1, layers * self.patch_samples)  # (BS)(LPP)

        counts = self.counts[inds].view(-1, layers * self.patch_samples).to(device=device)  # (BS)(LPP)
        features = self.features[inds].to(device=device)  # (BSL)(PP)F
        rows = torch.arange(0, distances.shape[0], device=points.device, dtype=torch.long)
        box_size = box_size.view(-1, 1)  # (BS)1

        cache = {'features': features, 'normals': normals.float(), 'inds': inds, 'counts': counts.unsqueeze(-1).float()}

        if mode == 'write' or mode == 'fuse':
            weights = torch.zeros_like(distances)  # (BS)(LPP)
            weights[distances < box_size] = 1.  # covered by the point box
            weights[rows, torch.argmin(distances, dim=-1)] = 1  # closest
            if mode == 'fuse':
                cache['read_weights'] = F.normalize(weights, p=1, dim=-1).unsqueeze(-1).float()  # (BSL)(PP)1

            cache['write_weights'] = F.normalize(weights, p=1, dim=-1).unsqueeze(-1).float()  # (BSL)(PP)1

        if return_read_weights:
            weights = torch.zeros_like(distances)  # (BS)(LPP)
            distances[counts == 0] = float('inf')  # (BS)(LPP)
            weights[distances < box_size] = 1.  # covered by the point box
            weights[rows, torch.argmin(distances, dim=-1)] = 1  # closest
            weights = F.normalize(weights, p=1, dim=-1).unsqueeze(-1).float()  # (BSL)(PP)1

            return cache, weights

        return cache

    def read_average(self, points, box_size, n_layers=0, mode='read'):  # (BS)4
        assert mode in ['read', 'fuse']

        cache, weights = self._projection(points, box_size, n_layers, mode=mode, return_read_weights=True)
        features, counts = cache['features'], cache['counts']

        # compute features
        features = features.view(points.shape[0], -1, self.channels)  # (BS)(LPP)F
        features = features * weights.view(points.shape[0], -1, 1)  # (BS)(LPP)F
        features = features.sum(dim=1)  # (BS)F

        # compute confidence
        counts = counts + 1
        counts[weights == 0] = 0
        confidence = counts.view(points.shape[0], -1).mean(-1).view(-1, 1)  # (BSL)(PP)1  ->  (BS)1

        return features, confidence, cache['normals'], cache

    def is_inside(self, points):
        mask = ~torch.isinf(points).any(-1)
        points = self._wc2lc(points[mask])
        points = points[:, 1:]  # remove batch dimension

        # 0.5 because of rounding
        mask[mask.clone()] = (points >= 0.5).all(-1) & (points <= (self.grid_resolution - 0.5)).all(-1)
        return mask

    def fuse(self, features, points, box_size, n_layers, moving_average_type=None, cache=None):
        assert features.shape[0] == points.shape[0] == box_size.shape[0]

        device = features.device
        layers = n_layers * 2 + 1
        if cache is None:
            cache = self._projection(points, box_size, n_layers, mode='fuse')

        # (BSL)4
        weights, texel_inds = cache['write_weights'], cache['inds']

        # create texel indices   # (BSL) -> (BSLPP) -> (BSLPPF)
        texel_inds = \
            texel_inds.repeat_interleave(self.patch_samples) * self.patch_samples + \
            torch.arange(self.patch_samples,
                         device=texel_inds.device,
                         dtype=texel_inds.dtype).long().repeat(texel_inds.shape[0])

        weights = weights.view(-1, 1)  # (BSLPP)1
        texels = features.repeat_interleave(layers * self.patch_samples, dim=0)  # (BSLPP)F
        texels = texels * weights  # (BSLPP)F

        assert texel_inds.shape[0] == texels.shape[0] == weights.shape[0]

        # sort indices
        inds, sorted_inds = texel_inds.sort()

        dhw_inds_unique, counts = torch.unique_consecutive(inds, return_counts=True)
        # del inds

        padded_unique_inds = torch.arange(dhw_inds_unique.shape[0], dtype=torch.long, device=device)
        padded_unique_inds = padded_unique_inds.repeat_interleave(counts, dim=0)

        # update
        tmp_weights = torch.zeros((dhw_inds_unique.shape[0], 1), dtype=weights.dtype, device=device)
        tmp_texture = torch.zeros((dhw_inds_unique.shape[0], self.channels), dtype=texels.dtype, device=device)

        tmp_weights = tmp_weights.index_add(0, padded_unique_inds, weights[sorted_inds])
        tmp_texture = tmp_texture.index_add(0, padded_unique_inds, texels[sorted_inds])
        del padded_unique_inds, texels

        # normalize
        nonzero_unique = tmp_weights.view(-1) != 0
        tmp_texture[nonzero_unique] = tmp_texture[nonzero_unique] / tmp_weights[nonzero_unique]

        del tmp_weights
        torch.cuda.empty_cache()
        rev_order = sorted_inds.sort()[1]
        altered_patches = tmp_texture.repeat_interleave(counts, dim=0)
        altered_patches = altered_patches[rev_order].view(-1, self.patch_samples, self.channels)

        nonzero_unique = nonzero_unique.repeat_interleave(counts)[rev_order].view(-1, self.patch_samples, 1)
        # nonzero_unique = nonzero_unique[..., 0].unsqueeze(-1)  # same weights for all channels

        del counts, rev_order

        # UPDATE state
        altered_patches = altered_patches[nonzero_unique.squeeze(-1)].view(-1, self.channels)
        prev_features = cache['features'][nonzero_unique.squeeze(-1)].view(-1, self.channels)
        tmp_w = cache['counts'].view(-1, self.patch_samples, 1).masked_select(nonzero_unique)
        # cache['counts'][nonzero_unique] = tmp_w + 1
        # texel_inds = texel_inds.view(-1, self.patch_samples, self.channels)[..., 0] / 3
        texel_inds = texel_inds.masked_select(nonzero_unique.view(-1))

        # update features and counts
        self.counts.view(-1)[texel_inds] = (tmp_w + 1).to(device=self.counts.device).detach()
        if moving_average_type == 'log':
            tmp_w = torch.log10(tmp_w + 1)

        new_features = (prev_features * tmp_w.view(-1, 1) + altered_patches) / (tmp_w.view(-1, 1) + 1)
        self.features.view(-1, self.channels)[texel_inds] = new_features.to(device=self.features.device).detach()

        # compute features
        cache['features'][nonzero_unique.squeeze(-1)] = new_features
        features = cache['features'].view(-1, layers * self.patch_samples, self.channels)  # (BS)(LPP)F
        features = features * cache['read_weights'].view(-1, layers * self.patch_samples, 1)
        features = features.sum(dim=1)  # (BS)F
        return features

    @staticmethod
    def _estimate_layer_locations(points, normals, n_layers):  # (BS)4, (BS)3
        layers = 2 * n_layers + 1
        offsets = np.array([i * np.sqrt(3) for i in range(-n_layers, n_layers + 1)])
        offsets = torch.from_numpy(offsets).to(dtype=normals.dtype, device=normals.device).view(1, -1, 1)  # 1L1

        normals = normals.repeat_interleave(layers, dim=0).view(-1, layers, 3)  # (BS)L3
        offsets = offsets * normals  # (BS)L3

        offsets = offsets.view(-1, 3)  # (BS)L3 -> (BSL)3
        points = points.repeat_interleave(layers, dim=0)  # (BSL)4

        points[:, 1:] += offsets
        return points  # (BSL)4

    def _estimate_surface_direction(self, points):  # (BS)4
        batches = points[:, 0]
        points = points[:, 1:] * self.geometry_upsampling_factor  # because of geometry upsampling factor

        resolution = self.grid_resolution * self.geometry_upsampling_factor
        dhw_inds, interpolation_weights = add_trilinear_neigh_points(points)  # BS,8
        d, h, w = dhw_inds[..., 0], dhw_inds[..., 1], dhw_inds[..., 2]  # BS,8
        batches = batches.view(-1, 1).repeat_interleave(8, dim=1).view(-1, 1).long()  # (BS)8
        inds = torch.cat((batches.view(-1, 1), d.view(-1, 1), h.view(-1, 1), w.view(-1, 1)), dim=1)  # (BS8)
        inds = self._bdhw2idx(inds, resolution)  # (BS8)

        normals = self.normals[inds].to(device=points.device)  # -1, 3
        normals = normals * interpolation_weights.view(-1, 1)  # -1, 3
        normals = normals.view(-1, 8, 3).sum(1)  # (BS),8,3 -> (BS)3
        normals = F.normalize(normals, dim=-1)

        return normals  # (BS)3
