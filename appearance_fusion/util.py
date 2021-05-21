import torch


def normalize_image(img):
    return 2 * (img.float() / 255.) - 1.0


def denormalize_image(img):
    img = 255. * (img + 1.) / 2.
    img = torch.round(img).to(dtype=torch.uint8)
    return img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_white_background(gt_image, image, mask, bg_value=1.):
    B, C, H, W = gt_image.shape
    mask = ~mask.view(B, H, W)
    gt_image.permute(0, 2, 3, 1)[mask] = bg_value
    image.permute(0, 2, 3, 1)[mask] = bg_value

