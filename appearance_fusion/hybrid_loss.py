import torch
import numpy as np

from torch import nn

from util import denormalize_image
from skimage.metrics import structural_similarity


class HybridLoss(nn.Module):
    def __init__(self, device, l1_weight=None, l2_weight=None):
        super().__init__()

        self.l1_loss = nn.L1Loss(reduction="none").to(device)
        self.l2_loss = nn.MSELoss(reduction="none").to(device)

        if l1_weight is None:
            l1_weight = 1.

        if l2_weight is None:
            l2_weight = 1.

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def _compute_psnr(self, gt_pixels, rendered_pixels):
        l2 = self.l2_loss(gt_pixels.float(), rendered_pixels.float()).mean()
        PSNR = 20 * np.log10(255.)
        if l2.item() != 0:
            PSNR -= 10 * np.log10(l2.item())

        return float(PSNR)

    @staticmethod
    def _compute_ssim(img1, img2, mask):
        i1 = img1.permute(0, 2, 3, 1).cpu().numpy()
        i2 = img2.permute(0, 2, 3, 1).cpu().numpy()
        ssim_img = []
        for i in range(img1.shape[0]):
            mssim, S = structural_similarity(i1[i], i2[i], data_range=2, multichannel=True, full=True)
            ssim_img.append(torch.from_numpy(S).permute(2, 0, 1))
        ssim_img = torch.stack(ssim_img).to(device=mask.device)
        ssim_img = ssim_img.permute(0, 2, 3, 1)[mask]
        ssim = ssim_img.mean()
        return ssim.item()
        # mask = mask.permute(0, 2, 3, 1).cpu().numpy()
        # ssim = compare_ssim(i1, i2, multichannel=True, full=True)
        # (_, channel, _, _) = img1.size()
        # # compute window
        # gauss = [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
        # gauss = torch.Tensor(gauss)
        # gauss = gauss / gauss.sum()
        #
        # _1D_window = gauss.unsqueeze(1)
        # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        # window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        # window = window.to(device=img1.device, dtype=img1.dtype)
        #
        # # compute ssim map
        # mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        # mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        #
        # mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        #
        # sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        # sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        # sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        #
        # # BCHW
        # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        # ssim_map = ssim_map.permute(0, 2, 3, 1)
        # ssim_map = ssim_map.masked_select(mask).view(-1, 3)
        # ssim_score = ssim_map.mean()
        # return ssim_score

    def __call__(self, gt_image, rendered_image, mask, omit_metrics=False):
        assert gt_image.shape == rendered_image.shape

        B, C, H, W = gt_image.shape
        mask = mask.view(B, H, W)
        gt_pixels = gt_image.permute(0, 2, 3, 1)[mask].view(-1, C)
        rendered_pixels = rendered_image.permute(0, 2, 3, 1)[mask].view(-1, C)

        l1_loss = self.l1_loss(gt_pixels, rendered_pixels).mean(dim=1).mean()
        l2_loss = self.l2_loss(gt_pixels, rendered_pixels).mean(dim=1).mean()

        loss_obj = self.l1_weight * l1_loss + self.l2_weight * l2_loss

        loss_logs = {
            'l1_loss': l1_loss.detach().item(),
            'l2_loss': l2_loss.detach().item(),
            'l_loss': loss_obj.detach().item(),
        }
        if not omit_metrics:
            loss_logs['SSIM'] = self._compute_ssim(gt_image, rendered_image, mask)
            gt_pixels = denormalize_image(gt_pixels)
            rendered_pixels = denormalize_image(rendered_pixels)
            loss_logs['PSNR'] = self._compute_psnr(gt_pixels, rendered_pixels)

        return loss_obj, loss_logs
