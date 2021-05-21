import json
from os import makedirs
from os.path import join, exists

import deep_surfel as dsurf
import torch
from torch.utils.data import Dataset, IterableDataset

from ._util import get_frame_ids, find_closest_frames, stack_frames
from .frame import Frame


class OneSceneIterableDataset(IterableDataset):
    def __init__(self, config, dir_path, scene, frame_ids, closest_frame_ids, mode):
        super(OneSceneIterableDataset).__init__()

        self.mode = mode
        self.config = config
        self.dir_path = dir_path
        self.frame_ids = frame_ids
        self.closest_frame_ids = closest_frame_ids
        self.scene = scene
        self.n_frames = len(frame_ids)
        self.offset = 0

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        for frame_id, closest_frame_id in zip(self.frame_ids, self.closest_frame_ids):
            frame = Frame(self.config, self.dir_path, self.scene, frame_id, self.mode)
            closest_frame = Frame(self.config, self.dir_path, self.scene, closest_frame_id, self.mode)
            yield frame(), closest_frame()

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        indices = list(range(worker_id, dataset.n_frames, worker_info.num_workers))
        dataset.frame_ids = [dataset.frame_ids[i] for i in indices]

    @staticmethod
    def collate_fn(batch):
        closest_frames_batch = [b[1] for b in batch]
        batch = [b[0] for b in batch]

        keys = batch[0].keys()
        for i in range(len(batch)):
            for key in keys:
                if key != 'frame_id':
                    batch[i][key] = batch[i][key].unsqueeze(0)
                    closest_frames_batch[i][key] = closest_frames_batch[i][key].unsqueeze(0)
        return batch, closest_frames_batch


class GeneralizationDataset(Dataset):

    def __init__(self, config, data_root, scene_ids, frame_ids):
        self.mode = 'train'
        self.config = config
        self.data_root = data_root
        self.scene_ids = scene_ids
        self.frame_ids = frame_ids
        self.n_samples = len(scene_ids)
        self.surfel_channels = config.surfel_channels

        self.batch_size = config.batch_size
        self.cache_dir = config.tmp_cache_dir
        self.current_batch_scene_ids = set()
        self.device = self.config.device
        # self.worker_id = 0

    def __len__(self):
        return self.n_samples

    def load_scene(self, scene_id):
        # worker_info = torch.utils.data.get_worker_info()
        scene_name = f'scene_{self.config.geometry_resolution}_{self.config.patch_resolution}.dsurf'

        if not exists(join(self.cache_dir, scene_id, 'cache.json')):  # first time
            makedirs(join(self.cache_dir, scene_id))
            dir_path = join(self.data_root, scene_id, self.mode)
            frame_ids = get_frame_ids(dir_path)
            closest_frame_ids = find_closest_frames(dir_path, frame_ids, self.config.dont_toggle_yz)
            with open(join(self.cache_dir, scene_id, 'cache.json'), 'w') as f:
                json.dump({'frame_ids': frame_ids, 'closest_frame_ids': closest_frame_ids, 'iterator': 0}, f)

            scene = dsurf.load(join(self.data_root, scene_id, scene_name))
            if scene.channels != self.surfel_channels:
                scene.reset(self.surfel_channels)
        else:
            scene = dsurf.load(join(self.cache_dir, scene_id, scene_name))
        scene.scene_id = scene_id

        return scene

    def load_frame(self, scene):
        dir_path = join(self.data_root, scene.scene_id, self.mode)
        with open(join(self.cache_dir, scene.scene_id, 'cache.json')) as f:
            cache = json.load(f)
            frame = Frame(self.config, dir_path, scene, cache['frame_ids'][cache['iterator']], self.mode)()
            closest_frame = Frame(self.config, dir_path, scene, cache['closest_frame_ids'][cache['iterator']],
                                  self.mode)()
            for key in frame.keys():
                if key != 'frame_id':
                    frame[key] = frame[key].unsqueeze(0)
                    closest_frame[key] = closest_frame[key].unsqueeze(0)

        return frame, closest_frame

    def __getitem__(self, idx):
        # print(f'{torch.utils.data.get_worker_info().id} loads {self.scene_ids[idx]}')
        print(f'loads {self.scene_ids[idx]}')
        scene = self.load_scene(self.scene_ids[idx])
        frame, closest_frame = self.load_frame(scene)
        return scene, frame, closest_frame

    @staticmethod
    def collate_fn(batch):
        scene = dsurf.stack([b[0] for b in batch])
        frame = stack_frames([b[1] for b in batch])
        closest_frame = stack_frames([b[2] for b in batch])

        return scene, frame, closest_frame
