#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
import os
from functools import partial
from typing import Callable, Optional, Any

import hydra
import lightning as L
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.pytorch import Callback
from torch.utils._pytree import tree_map
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    DistributedSampler,
    WeightedRandomSampler,
)

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.common.sampler import softmin
from uni2ts.data.loader import DataLoader

class DynamicWeightUpdateCallback(Callback):
    def __init__(self, dataset, temperature_schedule=None):
        """
        :param distances: distance to target dataset
        :param temperature_schedule: function that takes epoch and returns temperature
        """
        super().__init__()
        self.distances = self.get_weights(dataset)
        if temperature_schedule is None:
            def temperature_schedule(epoch):
                if epoch < 400:
                    return 0.0
                elif 400 <= epoch < 800:
                    return (epoch - 400) / 400.0
                else:
                    return 1.0
        self.temperature_schedule = temperature_schedule

    def on_train_epoch_start(self, trainer: L.Trainer, *args: Any, **kwargs: Any):
        """每个epoch开始时更新 sampler 的权重"""
        current_epoch = trainer.current_epoch
        new_weights = self.update_weights(current_epoch)
        
        # 获取训练数据加载器
        train_dataloader = trainer.train_dataloader
        sampler = train_dataloader.dataloader.sampler

        # 更新 sampler 的权重
        if isinstance(sampler, torch.utils.data.WeightedRandomSampler):
            sampler.weights = torch.tensor(new_weights)
            print(f"Epoch {current_epoch}: Updated sampler weights")

    def update_weights(self, epoch):
        """根据当前 epoch 动态计算新的权重"""
        # 这里假设使用温度衰减来调整权重，温度随 epoch 增加而变化
        temperature = self.temperature_schedule(epoch)
        new_weights = softmin(self.distances, temperature=temperature)
        return new_weights

    @staticmethod
    def get_weights(dataset):
        weights = []
        for sub_dataset in dataset.datasets:
            print(sub_dataset)
            if isinstance(sub_dataset, ConcatDataset):
                weights.extend(list(itertools.chain.from_iterable(d.weights for d in sub_dataset.datasets)))
            else:
                weights.extend(list(sub_dataset.weights))
        return weights


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    def get_dataloader(
        self,
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        try:
            weights = []
            for sub_dataset in dataset.datasets:
                print(sub_dataset)
                if isinstance(sub_dataset, ConcatDataset):
                    weights.extend(list(itertools.chain.from_iterable(d.weights for d in sub_dataset.datasets)))
                else:
                    weights.extend(list(sub_dataset.weights))
            self.distances = weights
            # weights = list(itertools.chain.from_iterable(d.weights for d in dataset.datasets))
            weights = softmin(weights, temperature=0.6)
            print("weights len: ", len(weights))
            print("dataset len: ", len(dataset))

            sampler = WeightedRandomSampler(weights, batch_size)
            print("Using WeightedRandomSampler with weights length: ", len(weights))
        except AttributeError:
            print("No weights found, using DistributedSampler")
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=None,
                    rank=None,
                    shuffle=shuffle,
                    seed=0,
                    drop_last=False,
                )
                if world_size > 1
                else None
            )
        
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return tree_map(
            partial(
                self.get_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),
                shuffle=self.cfg.val_dataloader.shuffle,
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches
        )


@hydra.main(version_base="1.3", config_name="varying_weights.yaml")
def main(cfg: DictConfig):
    cfg = cfg._set_flag("allow_objects", True) # for callbacks.1 interpolation
    
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    if cfg.compile:
        model.module.compile(mode=cfg.compile)
    trainer: L.Trainer = instantiate(cfg.trainer)
    train_dataset: Dataset = instantiate(cfg.data).load_dataset(
        model.train_transform_map
    )
    val_dataset: Optional[Dataset | list[Dataset]] = (
        tree_map(
            lambda ds: ds.load_dataset(model.val_transform_map),
            instantiate(cfg.val_data, _convert_="all"),
        )
        if "val_data" in cfg
        else None
    )
    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)
    # set weights update callback
    # if cfg.dynamic_weight_update:
    trainer.callbacks.append(
        DynamicWeightUpdateCallback(
            dataset=train_dataset
        )
    )

    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset),
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    load_dotenv()
    print("HF_DATASETS_IN_MEMORY_MAX_SIZE: ", os.getenv("HF_DATASETS_IN_MEMORY_MAX_SIZE"))
    main()
