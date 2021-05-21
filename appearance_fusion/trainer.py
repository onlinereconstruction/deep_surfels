from contextlib import contextmanager
from os.path import join

from tqdm import tqdm
import torch

from data.scene_iterators import OneSceneDataIterator, MultiSceneDataIterator
from evaluate import evaluate
from logger import Logger
from hybrid_loss import HybridLoss
from appearance_fusion_pipeline import ColorFusionPipeline, FeatureFusionPipeline, HybridFusionPipeline
from util import set_white_background, denormalize_image


class Trainer:
    def __init__(self, config, test_mode=False):
        self.config = config
        self.test_mode = test_mode

        self.logger = Logger(self.config, test_mode)
        if config.generalization_mode:
            self.data_iterator = MultiSceneDataIterator(self.config, config.data_root)
        else:
            self.data_iterator = OneSceneDataIterator(self.config, config.data_root)
        self.val_data_iterator = OneSceneDataIterator(self.config, config.test_data_root, 'val')
        self.test_data_iterator = OneSceneDataIterator(self.config, config.test_data_root, 'test')
        self.test_data_iterator.num_workers = 0
        self.test_data_iterator.batch_size = 1

        self._create_model()
        self.hybrid_loss = HybridLoss(self.config.device, self.config.l1_weight, self.config.l2_weight)

        if not test_mode:
            self.logger.save_config(self.pipeline)

        self.opt_step = 0

    def _create_model(self):
        if self.config.pipeline == 'feature':
            self.pipeline = FeatureFusionPipeline(self.config)
            self.optimizer = torch.optim.Adam(self.pipeline.parameters(), lr=self.config.lr)
        elif self.config.pipeline == 'hybrid':
            self.pipeline = HybridFusionPipeline(self.config)
            self.optimizer = torch.optim.Adam(self.pipeline.parameters(), lr=self.config.lr)
        elif self.config.pipeline == 'color':
            self.pipeline = ColorFusionPipeline(self.config)
            self.optimizer = None

        if self.logger.use_pretrained_model() and self.config.pipeline != 'color':
            self.iteration, self.epoch, self.optimizer = self.logger.load_model(self.pipeline)
        else:
            self.iteration, self.epoch = 0, 0
            self.pipeline.train()
            if self.config.use_cuda:
                self.pipeline.cuda()

        self.clear_optimizers()
        self.pipeline.zero_grad()

    def clear_optimizers(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def backward(self, loss):
        if self.optimizer is not None:
            loss = loss / self.config.opt_loops
            loss.backward()
            if (self.opt_step + 1) % self.config.opt_loops == 0:
                self.optimizer.step()
                self.clear_optimizers()
                self.opt_step = 0
            else:
                self.opt_step += 1

    @contextmanager
    def eval_mode(self, evaluation_mode):
        if evaluation_mode:
            with torch.no_grad():
                self.pipeline.eval()
                yield
                self.pipeline.train()
        else:
            yield

    def _fit_model(self, scene, frame, closest_frame):
        gt_image, rendered_img, mask = self.pipeline(scene, frame, closest_frame)
        loss_obj, loss_logs = self.hybrid_loss(gt_image, rendered_img, mask, True)
        if self.config.orthogonal_loss_weight != 0:
            orthogonal_loss = self.pipeline.get_orthogonal_loss()
            loss_obj += orthogonal_loss
            loss_logs['orth_loss'] = orthogonal_loss.detach().item()
        self.backward(loss_obj)
        return loss_logs

    def fit_model(self, scene, scene_id):
        optimization_loss_list, loss_list = [], []

        for i, (frame, closest_frame) in enumerate(self.data_iterator.get_frames(scene, scene_id)):
            loss_logs = self._fit_model(scene, frame, closest_frame)
            optimization_loss_list.append(loss_logs)

        self.logger.add_optimization_loss(optimization_loss_list, self.iteration)

    def evaluate_model(self, scene, scene_id, mode='val'):
        data_iterator = self.data_iterator if mode == 'val' else self.test_data_iterator
        loss_list, images, masks, frame_ids = self._evaluate_model(scene, scene_id, data_iterator)
        self.logger.add_loss(loss_list, images, masks, frame_ids, self.iteration, mode)

    def _evaluate_model(self, scene, scene_id, data_iterator):
        scene.reset()

        with self.eval_mode(True):
            i = -1
            for i, (frame, _) in tqdm(enumerate(data_iterator.get_frames(scene, scene_id, sort_frame_ids=True))):
                self.pipeline.fuse(scene, frame)

            assert i != -1, 'no frame has been fused'

            loss_list, images, masks, frame_ids = [], [], [], []
            for i, (frame, _) in enumerate(self.val_data_iterator.get_frames(scene, scene_id)):
                rendered_img = self.pipeline.render(scene, frame)
                set_white_background(frame['gt_image'], rendered_img, frame['mask'], bg_value=1.)
                _, loss_logs = self.hybrid_loss(frame['gt_image'], rendered_img, frame['mask'], False)

                loss_list.append(loss_logs)
                images.append(self.logger.concat_images(frame['gt_image'], rendered_img))
                masks.append(frame['mask'])
                frame_ids.append(join(scene_id, frame['frame_id']))

            return loss_list, torch.cat(images), torch.cat(masks), frame_ids

    def evaluate_generalization(self):
        loss_list, images, masks, frame_ids = [], [], [], []
        for scene, scene_id in self.test_data_iterator.get_scenes():
            _loss_list, _images, _masks, _frame_ids = self._evaluate_model(scene, scene_id, self.test_data_iterator)

            loss_list += _loss_list
            images.append(_images)
            masks.append(_masks)
            frame_ids += _frame_ids

        images = torch.cat(images)
        masks = torch.cat(masks)

        self.logger.add_loss(loss_list, images, masks, frame_ids, self.iteration, 'test')

    def test(self):
        for scene, scene_id in self.test_data_iterator.get_scenes():
            self.evaluate_model(scene, scene_id, 'test')
            if self.config.extract_meshes:
                import deep_surfel as dsurf
                mesh_path = join(self.logger.log_images_root, scene_id, f'mesh.ply')
                dsurf.export_mesh(mesh_path, scene, True, True, surfel_transformation=denormalize_image)
                print('Saved:', mesh_path)
                join(self.logger.log_images_root, scene_id)
            evaluate(join(self.logger.log_images_root, scene_id))

    def _train(self):
        while True:
            for scene, scene_id in self.data_iterator.get_scenes():
                for scene_iter in range(self.config.scene_iterations):
                    if scene_iter > 0:
                        scene.reset()

                    print('\n\n', 'iter %s/%s' % (self.iteration, self.config.max_iterations), scene_id, end='\t')

                    if self.config.pipeline != 'color':
                        self.fit_model(scene, scene_id)

                    if not self.config.no_validation:
                        if self.config.pipeline != 'color' and self.iteration % self.config.steps_til_val == 0:
                            self.evaluate_model(scene, scene_id)

                        if self.iteration % self.config.steps_til_test == 0:
                            self.evaluate_model(scene, scene_id, 'test')

                    if (self.iteration + 1) >= self.config.max_iterations:
                        self.logger.save_model(self.pipeline, self.optimizer, self.epoch, self.iteration)
                        return

                    if self.iteration % self.config.steps_til_ckpt == 0:
                        self.logger.save_model(self.pipeline, self.optimizer, self.epoch, self.iteration)

                    self.iteration += 1
            self.epoch += 1

    def _train_generalization(self):
        # assert self.config.pipeline != 'color'

        n_training_log_elements = 10
        optimization_loss_list, loss_list = [], []
        while True:
            for scene, frame, closest_frame in self.data_iterator.get_data():
                print('\n\n', 'iter %s/%s' % (self.iteration, self.config.max_iterations), scene.scene_id, end='\t')

                if self.config.pipeline != 'color':
                    optimization_loss_list.append(self._fit_model(scene, frame, closest_frame))
                    self.data_iterator.save_scene(scene)
                    # import deep_surfel as dsurf
                    # scenes = dsurf.split(scene)
                    # for i, sc in enumerate(scenes):
                    #     dsurf.export_as_off(f'{sc.scene_id.replace("/","_")}_{self.iteration}_4x4.off', sc, True, True)
                    #     print('Saved:', f'{sc.scene_id.replace("/","_")}_{self.iteration}_4x4.off')
                    if len(optimization_loss_list) > n_training_log_elements:  # train logs
                        self.logger.add_optimization_loss(optimization_loss_list, self.iteration)
                        optimization_loss_list = []

                if not self.config.no_validation and self.iteration % self.config.steps_til_test == 0:
                    self.evaluate_generalization()

                if (self.iteration + 1) >= self.config.max_iterations:
                    self.logger.save_model(self.pipeline, self.optimizer, self.epoch, self.iteration)
                    return

                if self.iteration % self.config.steps_til_ckpt == 0:
                    self.logger.save_model(self.pipeline, self.optimizer, self.epoch, self.iteration)

                self.iteration += 1
            self.epoch += 1

    def train(self):
        if self.config.generalization_mode:
            self._train_generalization()
        else:
            self._train()
