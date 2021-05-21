from os.path import join

import cv2
import numpy as np
import torch
from torch import nn

from ._util import uvd2xyz, get_ray_dirs, load_params
from util import normalize_image


class Frame:
    def __init__(self, config, dir_path, scene, frame_id, mode):
        self.mode = mode
        self.n_sub_pixels = config.n_sub_pixels
        self.image_resolution = config.image_resolution
        self.hr_image_res = self.image_resolution[0] * self.n_sub_pixels, self.image_resolution[1] * self.n_sub_pixels
        self.original_image_resolution = config.original_image_resolution
        self.upscale_pixel_plane = config.upscale_pixel_plane
        self.dont_toggle_yz = config.dont_toggle_yz
        self.superresolution_mode = config.superresolution_mode

        self.dir_path = dir_path
        self.scene = scene
        self.frame_id = frame_id

        self._load_data()
        self._select_masked_points()

    def _select_masked_points(self):
        if self.superresolution_mode and self.mode == 'train':
            self.gt_image = cv2.imread(join(self.dir_path, f'{self.frame_id}_img_hr.png'))
            self.gt_image = torch.from_numpy(cv2.cvtColor(self.gt_image, cv2.COLOR_BGR2RGB))
        self.gt_image = normalize_image(self.gt_image)
        self.image = normalize_image(self.image[self.points_mask])
        self.depth = self.depth[self.points_mask].unsqueeze(-1)

        self.points = self.points[self.points_mask]
        self.ray_dirs = self.ray_dirs[self.points_mask]
        self.pixel_size = self.pixel_size[self.points_mask]

    @torch.no_grad()
    def _load_data(self):
        img_path = join(self.dir_path, f'{self.frame_id}_img.png')
        depth_path = join(self.dir_path, f'{self.frame_id}_depth.exr')
        param_path = join(self.dir_path, f'{self.frame_id}_params.json')

        # load data
        self.K, self.extrinsics, self.cam2world = load_params(param_path, self.dont_toggle_yz)
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.depth.shape[-1] == 3 and len(self.depth.shape) == 3:
            self.depth = self.depth[..., 0]

        # resize images if needed
        if self.image_resolution != self.original_image_resolution:
            self.image = cv2.resize(self.image, self.image_resolution[::-1])  # cv2 takes WxH
            self.depth = cv2.resize(self.depth, self.image_resolution[::-1])
            # self.depth[np.isnan(self.depth)] = np.inf

            yscale = self.image_resolution[0] / self.original_image_resolution[0]
            xscale = self.image_resolution[1] / self.original_image_resolution[1]

            scaling = np.eye(3)
            scaling[0, 0] = xscale
            scaling[1, 1] = yscale
            self.K = np.dot(scaling, self.K)
            self.K = torch.from_numpy(self.K)

        # compute pixel frustum information
        uvd = self._get_uvd(self.image_resolution, self.depth)
        pixel_points = uvd2xyz(uvd, self.cam2world, self.K)
        self.mask = self.scene.is_inside(pixel_points).view(self.image_resolution)
        if self.superresolution_mode and self.mode == 'train':
            self.mask = self._upsample_mask(self.mask).to(dtype=self.mask.dtype)
        self.pixel_size, self.ray_dirs, self.points, self.points_mask = self._compute_3d_pixel(self.depth)

        # upsample data
        self.gt_image = torch.from_numpy(self.image)
        self.image = self.image.repeat(self.n_sub_pixels, axis=0).repeat(self.n_sub_pixels, axis=1)
        self.depth = self.depth.repeat(self.n_sub_pixels, axis=0).repeat(self.n_sub_pixels, axis=1)

        self.image, self.depth = torch.from_numpy(self.image), torch.from_numpy(self.depth)

    def _upsample_mask(self, mask):
        upsampler = nn.Upsample(scale_factor=self.n_sub_pixels, mode='nearest')
        mask = upsampler(mask.unsqueeze(0).unsqueeze(0).float())
        mask = mask.squeeze(0).squeeze(0)

        return mask

    def _compute_3d_pixel(self, depth):
        def _compute_3d_pixel_size(cam2world, K, depth, upscale_pixel_plane):
            uvt = np.flip(np.mgrid[0:depth.shape[0], 0:depth.shape[1]].astype(np.int32), axis=0)
            uvt = uvt.reshape((2, -1)).T
            d = depth[uvt[:, 1], uvt[:, 0]]

            uvdt_left = np.hstack((uvt, d[..., None]))
            uvdt_right = uvdt_left.copy()
            uvdt_right[:, 0] += 1

            xyz_left = uvd2xyz(torch.from_numpy(uvdt_left), cam2world, K)
            xyz_right = uvd2xyz(torch.from_numpy(uvdt_right), cam2world, K)

            _pixel_size = (xyz_right - xyz_left).numpy()
            _pixel_size[np.isinf(_pixel_size) | np.isnan(_pixel_size)] = 0
            _pixel_size = np.linalg.norm(_pixel_size, axis=-1)
            _pixel_size = upscale_pixel_plane * _pixel_size
            _pixel_size = _pixel_size.reshape(depth.shape)
            _pixel_size = _pixel_size.repeat(self.n_sub_pixels, axis=0).repeat(self.n_sub_pixels, axis=1)
            _pixel_size = _pixel_size / self.n_sub_pixels
            return torch.from_numpy(_pixel_size).view(*self.hr_image_res, 1).float()

        # hr_image_res = self.image_resolution[0] * self.n_sub_pixels, self.image_resolution[1] * self.n_sub_pixels
        pixel_size = _compute_3d_pixel_size(self.cam2world, self.K, depth, self.upscale_pixel_plane)

        # compute ray direction
        depth = depth.repeat(self.n_sub_pixels, axis=0).repeat(self.n_sub_pixels, axis=1)
        uvdt = self._get_uvd(self.hr_image_res, depth)
        ray_dirs = get_ray_dirs(uvdt, self.cam2world, self.K)
        ray_dirs = ray_dirs.view(*self.hr_image_res, 3).float()

        # compute 3d points
        points = uvd2xyz(uvdt, self.cam2world, self.K).view(*self.hr_image_res, 3)
        inside_mask = self.scene.is_inside(points).view(self.hr_image_res)

        return pixel_size, ray_dirs, points, inside_mask

    def _get_uvd(self, resolution, depth):
        uvt = np.flip(np.mgrid[0:resolution[0], 0:resolution[1]].astype(np.int32), axis=0)
        uvt = uvt.reshape((2, -1)).T
        d = depth[uvt[:, 1], uvt[:, 0]]

        uvd = torch.from_numpy(np.hstack((uvt / self.n_sub_pixels, d[..., None])))
        return uvd

    def __call__(self, *args, **kwargs):
        # import matplotlib.pyplot as plt
        # plt.imshow((denormalize_image(self.gt_image)).numpy())
        # plt.show()
        # plt.imshow((denormalize_image(self.gt_image)*self.mask.unsqueeze(-1)).numpy())
        # plt.show()
        to_ret = {
            # image resolution data
            'frame_id': self.frame_id,

            'gt_image': self.gt_image.permute(2, 0, 1).contiguous(),
            'depth': self.depth,
            'mask': self.mask.unsqueeze(0),

            # 'upsampled_mask': self.upsampled_mask.unsqueeze(0),

            # upsampled image resolution data
            'image': self.image,
            'points': self.points,
            'ray_dirs': self.ray_dirs,
            'pixel_size': self.pixel_size,
            'points_mask': self.points_mask.unsqueeze(0),
        }

        return to_ret
