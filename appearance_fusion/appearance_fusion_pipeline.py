from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from util import count_parameters
from layers import FeatureCompressor, FeatureRefiner, FeatureMapDecoder, NeuralFeatureFusion, \
    DeterministicRenderer


class AppearanceFusionPipeline(nn.Module, metaclass=ABCMeta):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_layers = config.n_layers

        self._build_net()
        print(self.get_parameters_statistics())

    @staticmethod
    def _features2image_format(features, mask):
        B, _, H, W = mask.shape  # B1HW
        read_features = torch.zeros((B, H, W, features.shape[-1]), device=features.device, dtype=features.dtype)
        read_features[mask.view(B, H, W)] = features
        return read_features.permute(0, 3, 1, 2).contiguous()

    @abstractmethod
    def _build_net(self):
        pass

    @abstractmethod
    def get_parameters_statistics(self):
        return ''

    @abstractmethod
    def fuse(self, scene, frame):
        pass

    @abstractmethod
    def render(self, scene, frame):
        pass

    @abstractmethod
    def get_orthogonal_loss(self):
        pass


class ColorFusionPipeline(AppearanceFusionPipeline):

    def __init__(self, config):
        super().__init__(config)
        self.n_channels = 3

    def _build_net(self):
        self.render_net = DeterministicRenderer(self.config)

    def _project(self, scene, frame):
        features, confidence, normals, cache = scene.read_average(frame['points'], frame['pixel_size'], self.n_layers)
        features = self._features2image_format(features, frame['points_mask'])
        return features, confidence, normals, cache

    def fuse(self, scene, frame):
        features = scene.fuse(frame['image'], frame['points'], frame['pixel_size'], self.n_layers)
        features = self._features2image_format(features, frame['points_mask'])
        return features

    def render(self, scene, frame):
        features = self._project(scene, frame)[0]
        return self.render_net(features, frame)

    def forward(self, scene, frame, closest_frame):
        features = self.fuse(scene, frame)
        rendered_image = self.render_net(features, frame)
        return frame['gt_image'], rendered_image, frame['mask']

    def get_parameters_statistics(self):
        return ''

    def get_orthogonal_loss(self):
        return 0


class FeatureFusionPipeline(AppearanceFusionPipeline):

    def __init__(self, config):
        self._init_channels(config)

        super().__init__(config)

    def _init_channels(self, config):
        res = config.image_resolution
        self.image_res = config.image_resolution
        self.surfel_channels = config.surfel_channels
        self.channels = config.channels
        self.meta_features = config.meta_features

        self.upsampled_image_res = res[0] * config.n_sub_pixels, res[1] * config.n_sub_pixels

        self.in_ch = 3 + self.surfel_channels + self.meta_features
        self.embeddings_length = 35
        self.feature_channels = self.embeddings_length * 2
        self.refined_features_channels = self.feature_channels + self.meta_features

    def _build_net(self):
        self.embedding_layer = nn.Linear(self.in_ch, self.embeddings_length, bias=True)
        self.feature_compressor = FeatureCompressor(self.feature_channels, self.channels, self.config.device)

        self.fusion_net = NeuralFeatureFusion(self.embeddings_length, self.feature_channels)
        self.feature_refiner = FeatureRefiner(self.refined_features_channels, self.image_res)
        self.feature_map_decoder = FeatureMapDecoder(self.refined_features_channels)

        self.sum_pool_mask_filters = torch.ones(
            (1, 1, self.config.n_sub_pixels, self.config.n_sub_pixels),
            device=self.config.device)

    def get_parameters_statistics(self):
        return '\n'.join([
            f'Total number of parameters: {count_parameters(self)}',
            f'\tembedding_layer: {count_parameters(self.embedding_layer)}',
            f'\tfeature_compressor: {count_parameters(self.feature_compressor)}',
            f'\tfusion_net: {count_parameters(self.fusion_net)}',
            f'\tfeature_refiner: {count_parameters(self.feature_refiner)}',
            f'\tfeature_map_decoder: {count_parameters(self.feature_map_decoder)}',
        ])

    def _stack_features(self, features, frame, confidence=None, normals=None):
        if self.config.use_confidence:
            features = torch.cat((features, confidence), dim=-1)

        if self.config.use_ray_weights:
            # ray_weights = (normals * frame['ray_dirs']).sum(-1, keepdim=True)  # cosine angle
            # features = torch.cat((features, ray_weights), dim=-1)
            features = torch.cat((features, normals), dim=-1)

            # import matplotlib.pyplot as plt
            # img = self._features2image_format((ray_weights+1)/2., frame['points_mask'])
            # # plt.imshow(img.view(1, 2*512, 2*512).permute(1, 2, 0).cpu().numpy())
            # plt.imshow(img.view(2*512, 2*512).cpu().numpy())
            # plt.colorbar()
            # plt.show()

        if self.config.use_ray_directions:
            features = torch.cat((features, frame['ray_dirs']), dim=-1)

        if self.config.use_depth:
            features = torch.cat((features, frame['depth']), dim=-1)

        return features

    def fuse(self, scene, frame):
        features, confidence, normals, cache = scene.read_average(frame['points'], frame['pixel_size'], self.n_layers,
                                                                  'fuse')

        # stack features
        features = self._stack_features(features, frame, confidence, normals)
        features = torch.cat((features, frame['image']), dim=-1)

        features = self.embedding_layer(features)

        # create surfel features
        features = self._features2image_format(features, frame['points_mask'])
        features = self.fusion_net(features, frame)
        features = features.permute(0, 2, 3, 1)[frame['points_mask'].permute(0, 2, 3, 1).squeeze(-1)]
        features = self.compress_features(features, frame)

        # update scene
        features = scene.fuse(features, frame['points'], frame['pixel_size'], self.n_layers, cache=cache)

        return features, confidence, normals, cache

    def compress_features(self, features, frame):
        features = self.feature_compressor(features)
        return features

    def extract_features(self, features):
        features = self.feature_compressor.extract(features)
        return features

    def forward(self, scene, frame, closest_frame):
        features, confidence, normals, cache = self.fuse(scene, frame)
        rendered_image = self._render(frame, features, confidence, normals, 'train')
        gt_image, mask = frame['gt_image'], frame['mask']
        if self.config.use_closest_frame_loss:
            features, confidence, normals, _ = scene.read_average(closest_frame['points'], closest_frame['pixel_size'],
                                                                  self.n_layers, 'read')
            # remove features where texels are empty
            confidence_mask = (confidence > 1/self.config.patch_resolution**2).view(-1)
            features = features[confidence_mask]
            confidence = confidence[confidence_mask]
            normals = normals[confidence_mask]

            closest_frame['ray_dirs'] = closest_frame['ray_dirs'][confidence_mask]
            closest_frame['depth'] = closest_frame['depth'][confidence_mask]
            confidence_mask = self._features2image_format(confidence_mask.view(-1, 1), closest_frame['points_mask'])
            closest_frame['points_mask'] &= confidence_mask

            closest_rendered_image = self._render(closest_frame, features, confidence, normals, 'train')

            gt_image = torch.cat((gt_image, closest_frame['gt_image']), dim=0)
            rendered_image = torch.cat((rendered_image, closest_rendered_image), dim=0)
            mask = torch.cat((mask, closest_frame['mask']), dim=0)

        return gt_image, rendered_image, mask

    def avg_pooling(self, feature_map, frame, mode='test'):
        if (mode == 'train' and self.config.superresolution_mode) or self.config.n_sub_pixels == 1:
            return feature_map

        feature_map = nn.functional.avg_pool2d(feature_map, self.config.n_sub_pixels, stride=self.config.n_sub_pixels)
        feature_map = (self.config.n_sub_pixels ** 2) * feature_map
        counts = frame['points_mask'].float()
        counts = nn.functional.conv2d(counts, self.sum_pool_mask_filters, stride=self.config.n_sub_pixels).clamp(min=1)
        feature_map = feature_map / counts
        return feature_map

    def _render(self, frame, features, confidence, normals, mode='test'):
        features = self.extract_features(features)
        features = self._stack_features(features, frame, confidence, normals)

        # downsample features
        features = self._features2image_format(features, frame['points_mask'])
        features = self.avg_pooling(features, frame, mode)

        # render features
        features = self.feature_refiner(features)
        rendered_image = self.feature_map_decoder(features)

        return rendered_image

    def render(self, scene, frame):
        features, confidence, normals, cache = scene.read_average(frame['points'], frame['pixel_size'], self.n_layers,
                                                                  'read')
        rendered_image = self._render(frame, features, confidence, normals)

        return rendered_image

    def get_orthogonal_loss(self):
        return self.config.orthogonal_loss_weight*self.feature_compressor.get_orthogonal_loss()


class HybridFusionPipeline(FeatureFusionPipeline):
    def __init__(self, config):
        self.n_channels = config.channels + 3
        self.feature_channels = config.channels

        super().__init__(config)

    def _init_channels(self, config):
        FeatureFusionPipeline._init_channels(self, config)
        self.refined_features_channels += 3

    def compress_features(self, features, frame):
        features = FeatureFusionPipeline.compress_features(self, features, frame)
        features = torch.cat((frame['image'], features), dim=-1)
        return features

    def extract_features(self, features):
        color_features, features = features[..., :3], features[..., 3:]
        features = FeatureFusionPipeline.extract_features(self, features)
        features = torch.cat((features, color_features), dim=-1)
        return features
