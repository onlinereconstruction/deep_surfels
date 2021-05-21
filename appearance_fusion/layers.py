import torch
from torch import nn


class FusionBlock(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels, activation_fn, input_resolution=None, dropout_prob=0):
        super().__init__()

        net = []
        for layer in range(n_layers):
            if layer == 0:
                net.append(nn.Conv2d(in_channels, out_channels, (3, 3), padding=1))
            else:
                net.append(nn.Conv2d(out_channels, out_channels, (3, 3), padding=1))

            if input_resolution is not None:
                net.append(nn.LayerNorm([out_channels, input_resolution[0], input_resolution[1]]))

            net.append(activation_fn())
            if dropout_prob != 0:
                net.append(nn.Dropout2d(p=dropout_prob))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class DeterministicRenderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.scale_factor = 1 / self.config.n_sub_pixels

        self._build_net()

    def _build_net(self):
        self.render_net = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

    def forward(self, feature_map, frame):
        feature_map = self.render_net(feature_map)
        return feature_map


class NeuralFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self._build_net(in_channels, out_channels)

    def _build_net(self, in_channels, out_channels):
        self.block = nn.Sequential(
            FusionBlock(4, in_channels, out_channels, nn.LeakyReLU, None, dropout_prob=.2),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
        )

    def forward(self, feature_map, frame):
        feature_map = self.block(feature_map)
        return feature_map


class FeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._build_net()
        self.l2_loss = nn.MSELoss().to(device)
        self.identity_matrix = torch.eye(self.in_channels, device=device)

    def _build_net(self):
        self.weight = nn.Linear(self.in_channels, self.out_channels, bias=False).weight

    def forward(self, x):
        assert x.shape[-1] == self.in_channels

        return x.mm(self.weight.t())

    def extract(self, x):
        assert x.shape[-1] == self.out_channels

        x = x.mm(self.weight)

        return x

    def get_orthogonal_loss(self):
        return self.l2_loss(torch.matmul(self.weight.t(), self.weight), self.identity_matrix)


class FeatureRefiner(nn.Module):
    def __init__(self, in_channels, input_resolution):
        super().__init__()

        self.in_channels = in_channels
        self.input_resolution = input_resolution

        self._build_net()

    def _build_net(self):
        self.block1 = nn.Sequential(
            FusionBlock(2, self.in_channels, self.in_channels, nn.LeakyReLU, None, dropout_prob=.2),
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            # nn.Hardtanh(),
            nn.Dropout2d(p=.2))
        self.block_activation = nn.LeakyReLU()
        self.block2 = nn.Sequential(
            FusionBlock(3, self.in_channels, self.in_channels, nn.LeakyReLU, None, dropout_prob=.2),
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1)
        )

    def forward(self, x):
        x = self.block_activation(x + self.block1(x))
        x = self.block2(x)
        return x


class FeatureMapDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 3

        self._build_net()

    def _build_net(self):
        self.net = []

        i = 1
        channels = self.in_channels
        while channels != self.out_channels:
            next_ch = max(self.out_channels, channels // (2 ** i))
            self.net.append(nn.Linear(channels, next_ch))
            channels = next_ch
            if channels != self.out_channels:
                self.net.append(nn.LeakyReLU())
            else:  # last layer
                self.net.append(nn.Hardtanh())

            i += 1
        self.net = nn.Sequential(*self.net)

    def forward(self, image):
        B, C, H, W = image.shape
        image = image.view(B, C, -1).permute(0, 2, 1)
        image = self.net(image)
        image = image.permute(0, 2, 1).contiguous().view(B, self.out_channels, H, W)
        return image
