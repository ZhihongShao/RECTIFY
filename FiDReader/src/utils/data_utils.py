#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""
import json
import pickle

import math
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import List, Iterator, Callable, Tuple

from .log_utils import log


def read_serialized_data_from_files(paths: List[str], num_shards=1, shard_id=0) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            log("Reading file {}".format(path))
            data = pickle.load(reader)
            if num_shards <= 1:
                results.extend(data)
            else:
                results.extend([item for idx, item in enumerate(data) if idx % num_shards == shard_id])
            log("Aggregated data size: {}".format(len(results)))
    log("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            log("Reading file {}".format(path))
            data = json.load(f)
            results = data
            log("Aggregated data size: {}".format(len(results)))
    return results


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: int = None, rank: int = None, shuffle: bool = None, seed: int = 0, drop_last: bool = False, duplicate_within_proc: bool = False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.duplicate_within_proc = duplicate_within_proc

    def __iter__(self):
        if not self.drop_last and self.duplicate_within_proc:
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            else:
                indices = list(range(len(self.dataset)))  # type: ignore

            # add extra samples to make it evenly divisible
            indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
            if len(indices) < self.num_samples:
                indices = (indices * math.ceil(self.num_samples / len(indices)))[: self.num_samples]
            assert len(indices) == self.num_samples

            return iter(indices)
        else:
            return super().__iter__()


def make_data_loader(dataset: Dataset, is_train: bool, batch_size: int, drop_last: bool, shuffle_seed: int = 0, collate_fn=None):
    sampler = CustomDistributedSampler(dataset, shuffle=is_train, seed=shuffle_seed, drop_last=drop_last, duplicate_within_proc=(not is_train))
    if collate_fn is None:
        collate_fn = lambda x: x
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return dataloader
