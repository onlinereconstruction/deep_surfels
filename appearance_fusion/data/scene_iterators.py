import json
from os.path import join
from random import shuffle, randint

import deep_surfel as dsurf
from torch.utils.data import DataLoader

from ._util import get_frame_ids, get_scene_ids, find_closest_frames, prepare_frame_single_batch, frame2device
from .datasets import OneSceneIterableDataset, GeneralizationDataset


class SceneIterator:
    def __init__(self, config, data_root, mode='train'):
        assert mode in ['train', 'test', 'val']

        self.mode = mode
        self.batch_size = config.batch_size
        self.config = config
        self.surfel_channels = config.surfel_channels
        self.data_root = data_root
        self.num_workers = max(0, config.num_workers)
        self.device = config.device
        self.cache_dir = config.tmp_cache_dir
        self.scene_ids = get_scene_ids(data_root)


class MultiSceneDataIterator(SceneIterator):
    def __init__(self, config, data_root):
        super().__init__(config, data_root, 'train')

        self.frame_ids = get_frame_ids(join(self.data_root, self.scene_ids[0], self.mode))

        ds = GeneralizationDataset(config, data_root, self.scene_ids, self.frame_ids)
        self.data_iterator = DataLoader(ds, num_workers=self.num_workers,  # worker_init_fn=ds.worker_init_fn,
                                        collate_fn=ds.collate_fn,
                                        batch_size=self.batch_size, drop_last=True)

    def save_scene(self, scene):
        scenes = dsurf.split(scene)
        for scene in scenes:
            print(f'Saving {scene.scene_id}')
            dir_root = join(self.config.tmp_cache_dir, scene.scene_id)
            cache_file = join(dir_root, 'cache.json')
            scene_file = join(dir_root, f'scene_{self.config.geometry_resolution}_{self.config.patch_resolution}.dsurf')

            with open(cache_file, 'r') as f:  # read cache
                cache = json.load(f)

            # random reset
            if randint(1, 50) == 1:
                scene.reset()

            # update cache
            cache['iterator'] += 1
            if cache['iterator'] == len(cache['frame_ids']):
                cache['iterator'] = 0
                scene.reset()

            # save cache
            with open(cache_file, 'w') as f:
                json.dump(cache, f)

            dsurf.save(scene_file, scene)

    def get_data(self):
        for batch in self.data_iterator:
            frame2device(batch[1], self.device)
            frame2device(batch[2], self.device)
            yield batch


class OneSceneDataIterator(SceneIterator):
    def __init__(self, config, data_root, mode='train'):
        super().__init__(config, data_root, mode)
        self.batch_size = 1

    def get_scenes(self):
        for scene_ind in range(len(self.scene_ids)):
            scene_name = f'scene_{self.config.geometry_resolution}_{self.config.patch_resolution}.dsurf'
            scene_id = self.scene_ids[scene_ind]
            scene = dsurf.load(join(self.data_root, scene_id, scene_name))
            if scene.channels != self.surfel_channels:
                scene.reset(self.surfel_channels)

            yield scene, scene_id

    def get_frames(self, scene, scene_id, sort_frame_ids=False):
        dir_path = join(self.data_root, scene_id, self.mode)

        frame_ids = get_frame_ids(dir_path)
        if sort_frame_ids:
            frame_ids = sorted(frame_ids, key=int)
        else:
            shuffle(frame_ids)

        closest_frame_ids = find_closest_frames(dir_path, frame_ids, self.config.dont_toggle_yz)
        ds = OneSceneIterableDataset(self.config, dir_path, scene, frame_ids, closest_frame_ids, self.mode)
        data_iterator = DataLoader(ds, num_workers=self.num_workers, worker_init_fn=ds.worker_init_fn,
                                   collate_fn=ds.collate_fn,
                                   batch_size=self.num_workers + 1)
        for batch, closest_frames_batch in data_iterator:
            for frame, closest_frame in zip(batch, closest_frames_batch):
                prepare_frame_single_batch(frame, self.device)
                prepare_frame_single_batch(closest_frame, self.device)
                yield frame, closest_frame
