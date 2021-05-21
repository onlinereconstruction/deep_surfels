import os
from os.path import join

import configargparse
import torch

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# model parameters
p.add_argument('--pipeline', default='color', type=str, required=False,
               choices=['color', 'feature', 'hybrid'],
               help='Which type of pipeline to use.')
p.add_argument('--n_layers', default=0, type=int, required=False,
               help='How many layers to select (deprecated).')
p.add_argument('--use_ray_weights', default=False, required=False, action='store_true',
               help='Whether to use ray weights.')
p.add_argument('--use_ray_directions', default=False, required=False, action='store_true',
               help='Whether to use ray directions as features.')
p.add_argument('--use_depth', default=False, required=False, action='store_true',
               help='Whether to depth map features.')
p.add_argument('--use_confidence', default=False, required=False, action='store_true',
               help='Whether to the confidence map.')
p.add_argument('--use_closest_frame_loss', default=False, required=False, action='store_true',
               help='Whether to use the closet frame loss regularization.')

p.add_argument('--weights_warp', default=None, required=False, choices=[None, 'log'],
               help='Transformation over the running average')

# data loading parameters
p.add_argument('--data_root', default='../data_prep/data_samples', required=False,
               help='Path to directory with raw training data.')
p.add_argument('--test_data_root', default=None, required=False,
               help='Path to directory with raw test data. By default it uses `data_root`.')
p.add_argument('--num_workers', default=4, type=int, required=False,
               help='Number of data loader workers.')
p.add_argument('--batch_size', default=1, type=int, required=False, help='Batch size.')
p.add_argument('--image_resolution', default=(312, 312), type=int, nargs='+', required=False, help='Image resolution.')
p.add_argument('--original_image_resolution', default=(312, 312), type=int, nargs='+', required=False,
               help='Original image resolution.')
p.add_argument('--geometry_resolution', default=64, type=int, required=False,
               help='DeepSurfel grid resolution.')
p.add_argument('--dont_toggle_yz', action='store_true', default=False,
               help='Dont toggle YZ axis (blender dataset compatibility).')
p.add_argument('--no_validation', action='store_true', default=False, help='Disable validation.')
p.add_argument('--generalization_mode', action='store_true', default=False, help='Generalization experiment.')
p.add_argument('--superresolution_mode', action='store_true', default=False, help='Superresolution experiment.')

# pipeline configuration
p.add_argument('--patch_resolution', default=4, type=int, required=False, help='Texture size.')
p.add_argument('--n_sub_pixels', default=1, type=int, help='Pixel oversampling factor.')
p.add_argument('--channels', default=3, type=int, required=False, help='Number of feature channels.')
p.add_argument('--upscale_pixel_plane', default=1, type=int, help='Pixel size.')

# computation flags
p.add_argument('--use_cuda', action='store_true', default=False, help='Whether to run on cpu or gpu.')
p.add_argument('--extract_meshes', action='store_true', default=False,
               help='Whether to extract DeepSurfel mesh at inference time.')

# optimization
p.add_argument('--max_iterations', default=10000, type=int, required=False, help='Maximum number of iterations.')
p.add_argument('--scene_iterations', type=int, default=1, help='How many training iterations per scene.')
p.add_argument('--opt_loops', type=int, default=1, help='After how many iterations to perform update.')
p.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
p.add_argument('--l1_weight', default=1, type=float, required=False, help='Weight for the l1 loss.')
p.add_argument('--l2_weight', default=0.5, type=float, required=False, help='Weight for the l2 loss.')
p.add_argument('--orthogonal_loss_weight', default=0, type=float, required=False,
               help='Weight for the orthogonal loss regularization.')

# checkpoints
p.add_argument('--logging_root_dir', type=str, default='./logs', required=False,
               help='A directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--tmp_cache_dir', type=str, default='./tmp', required=False,
               help='A directory where to dump intermediate scenes.')
p.add_argument('--experiment_name', type=str, default='debugging', required=False,
               help='Experiment name.')
p.add_argument('--overwrite_experiment', action='store_true', default=False,
               help='The training is continued by default.')
p.add_argument('--checkpoint_path', type=str, default=None, help='Load pretrained model.')

# logs
p.add_argument('--steps_til_val', type=int, default=5,
               help='The number of iterations until the validation set is processed.')
p.add_argument('--steps_til_test', type=int, default=5,
               help='The number of iterations until the test set is processed.')
p.add_argument('--steps_til_ckpt', type=int, default=50,
               help='The number of iterations until the checkpoint is saved.')


def get_config():
    config = p.parse_args()

    config.use_cuda = config.use_cuda and torch.cuda.is_available()
    config.device = 'cuda' if config.use_cuda else 'cpu'

    config.image_resolution = tuple(config.image_resolution)
    config.original_image_resolution = tuple(config.original_image_resolution)
    config.surfel_channels = config.channels
    config.meta_features = get_num_meta_features(config)
    config.tmp_cache_dir = join(config.tmp_cache_dir, config.experiment_name)
    if config.test_data_root is None:
        config.test_data_root = config.data_root

    if config.pipeline == 'hybrid':
        config.surfel_channels = config.surfel_channels + 3

    if 'TMPDIR' in os.environ:  # overwrite on the server
        config.tmp_cache_dir = os.environ['TMPDIR']

    config.commit_head = _get_git_commit_head()
    return config


def _get_git_commit_head():
    try:
        import subprocess
        head = subprocess.check_output("git rev-parse HEAD", stderr=subprocess.DEVNULL, shell=True)
        return head.decode('utf-8').strip()
    except:
        return ''


def get_num_meta_features(config):
    n_channels = 0

    if config.use_ray_weights:
        n_channels += 3

    if config.use_ray_directions:
        n_channels += 3

    if config.use_depth:
        n_channels += 1

    if config.use_confidence:
        n_channels += 1

    return n_channels
