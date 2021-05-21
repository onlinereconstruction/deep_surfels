from glob import glob
from os.path import basename, join

import configargparse
import cv2
import numpy as np
import torch
import yaml

from hybrid_loss import HybridLoss
from util import normalize_image


def load_data(root_dir, frame_id):
    mask_path = join(root_dir, f'{frame_id}_mask.npy')
    img_path = join(root_dir, f'{frame_id}_gt.png')
    rendered_path = join(root_dir, f'{frame_id}_rendered.png')

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    rendered_image = cv2.cvtColor(cv2.imread(rendered_path), cv2.COLOR_BGR2RGB)
    mask = np.load(mask_path)

    return image, rendered_image, mask


def evaluate(root_dir):
    hybrid_loss_obj = HybridLoss('cpu')
    image_ids = glob(join(root_dir, '*.png'))
    image_ids = list(map(lambda x: basename(x).split('_')[0], image_ids))
    image_ids = list(set(image_ids))

    images, rnd_imgs, masks = [], [], []
    for image_id in image_ids:
        data = load_data(root_dir, image_id)
        # if '00000' == image_id:
        #     break
        # print(image_id)
        images.append(data[0])
        rnd_imgs.append(data[1])
        masks.append(data[2])
        # break

    images = normalize_image(torch.from_numpy(np.stack(images))).permute(0, 3, 1, 2)
    rnd_imgs = normalize_image(torch.from_numpy(np.stack(rnd_imgs))).permute(0, 3, 1, 2)
    masks = torch.from_numpy(np.stack(masks))

    loss = hybrid_loss_obj(images, rnd_imgs, masks)[1]

    print(loss)
    with open(join(root_dir, 'results.yml'), 'w') as outfile:
        yaml.dump(loss, outfile, default_flow_style=False)


if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, required=True,
                   help='Directory that contains masks, rendered and ground truth images.')

    config = p.parse_args()
    evaluate(config.root_dir)
