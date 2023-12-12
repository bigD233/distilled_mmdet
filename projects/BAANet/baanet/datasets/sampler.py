import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler
import random
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class CustomKAISTSampler(Sampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 oversample_factor=2, 
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        # 获取所有样本的索引
        self.indices = list(range(len(dataset)))

        # 根据需要进行过采样的文件名和倍数生成过采样的样本索引
        self.oversample_factor = oversample_factor
        self.oversample_inds = []
        data_list = dataset.load_data_list()
        for i in range(len(data_list)):
            if any(substring in data_list[i]['img_path'] for substring in ['set00', 'set01', 'set02']):
                self.oversample_inds.extend([i] *(oversample_factor - 1))
        
        self.indices = self.indices + self.oversample_inds

        if self.round_up:
            self.num_samples = math.ceil(len(self.indices) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.indices) - rank) / world_size)
            self.total_size = len(self.indices)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            random.seed(self.seed + self.epoch)
            indices = random.sample(self.indices, len(self.indices))
            
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch