import shutil
from glob import glob
from os import makedirs
from os.path import join, exists, isdir, dirname

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from util import denormalize_image


class Logger:
    def __init__(self, config, test_mode=False):
        self.config = config
        self.test_mode = test_mode

        self.writer = None

        self.logging_root = join(config.logging_root_dir, config.experiment_name)
        self.log_images_root = join(self.logging_root, 'images')
        if not glob(join(self.logging_root, 'checkpoints', '*')):
            self.overwrite_experiment = True

        self.tmp_cache_dir = config.tmp_cache_dir

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is not None and isdir(self.checkpoint_path):
            self.checkpoint_path = sorted(glob(join(self.checkpoint_path, "*.pth")))[-1]

        if not test_mode:
            self._prepare_directories()

    @staticmethod
    def cond_mkdir(dir_path):
        if not exists(dir_path):
            makedirs(dir_path)

    def _prepare_directories(self):

        def cond_rmdir(dir_path):
            if isdir(dir_path):
                shutil.rmtree(dir_path, ignore_errors=False, onerror=None)

        if self.config.overwrite_experiment:
            cond_rmdir(self.logging_root)
            cond_rmdir(self.tmp_cache_dir)

        self.ckpt_dir = join(self.logging_root, 'checkpoints')
        events_dir = join(self.logging_root, 'events')

        self.cond_mkdir(self.tmp_cache_dir)
        self.cond_mkdir(self.logging_root)
        self.cond_mkdir(self.log_images_root)
        self.cond_mkdir(self.ckpt_dir)
        self.cond_mkdir(events_dir)

        self.writer = SummaryWriter(events_dir)

    def save_config(self, model=None):
        with open(join(self.logging_root, "params.txt"), "w") as out_file:
            out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(self.config).items()]))

        if model is not None:
            with open(join(self.logging_root, "model.txt"), "w") as out_file:
                out_file.write(f'{model.get_parameters_statistics()}\n\n{str(model)}')

    def use_pretrained_model(self):
        return self.checkpoint_path is not None

    def load_model(self, model):
        assert self.checkpoint_path is not None

        print("Loading model from %s" % self.checkpoint_path)

        whole_dict = torch.load(self.checkpoint_path)
        state = model.state_dict()
        state.update(whole_dict['model'])
        model.load_state_dict(state)
        if self.config.use_cuda:
            model.cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        optimizer.load_state_dict(whole_dict['optimizer'])

        return whole_dict['iteration'], whole_dict['epoch'], optimizer

    def save_model(self, model, optimizer, epoch, iteration):
        path = join(self.ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iteration))
        print('\nSaving model (iter=%i)\t%s' % (iteration, path))

        whole_dict = {
            'model': model.state_dict(),
            'iteration': iteration,
            'epoch': epoch
        }
        if optimizer:
            whole_dict.update({'optimizer': optimizer.state_dict()})
        torch.save(whole_dict, path)

    def add_optimization_loss(self, loss_list, iteration):
        self._log(loss_list, 'optimization', iteration)

    def add_loss(self, loss_list, image_list, masks, frame_ids, iteration, mode):
        self._log(loss_list, mode, iteration, image_list, masks, frame_ids)

    def _log(self, loss_list, mode, iteration, images=None, masks=None, frame_ids=None):
        if not self.test_mode:
            for key in loss_list[0].keys():
                self.writer.add_scalar(f'{mode}/{key}', np.mean([l[key] for l in loss_list]), iteration)

        if images is not None:
            if mode == 'test':  # save rendered images
                gt_images = images[:, :, :images.shape[2] // 2, :].permute(0, 2, 3, 1).numpy()
                rendered_images = images[:, :, images.shape[2] // 2:, :].permute(0, 2, 3, 1).numpy()

                dirs = list(set([dirname(join(self.log_images_root, f'{fid}_gt.png')) for fid in frame_ids]))
                for dir_path in dirs:
                    self.cond_mkdir(dir_path)

                for i in range(images.shape[0]):
                    cv2.imwrite(join(self.log_images_root, f'{frame_ids[i]}_gt.png'),
                                cv2.cvtColor(gt_images[i], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(join(self.log_images_root, f'{frame_ids[i]}_rendered.png'),
                                cv2.cvtColor(rendered_images[i], cv2.COLOR_RGB2BGR))
                    np.save(join(self.log_images_root, f'{frame_ids[i]}_mask.npy'), masks[i].cpu().numpy())

            # select maximum 5 images to display
            inds = torch.from_numpy(np.random.choice(images.shape[0], min(5, images.shape[0]), replace=False)).long()
            images = images[inds]
            images = torch.cat([images[i] for i in range(images.shape[0])], dim=-1)  # concat width
            if not self.test_mode:
                self.writer.add_image(mode, images, iteration)

    @staticmethod
    def concat_images(gt_img, rendered_img):
        gt_img = denormalize_image(gt_img).detach().clone()  # * mask
        rendered_img = denormalize_image(rendered_img).detach().clone()  # * mask

        # mask = (~mask * 255).to(dtype=gt_img.dtype)
        # gt_img |= mask  # gt_img[mask] = 255
        # rendered_img |= mask  # rendered_img[mask] = 255

        images = torch.cat((gt_img, rendered_img), dim=2).cpu()
        return images
