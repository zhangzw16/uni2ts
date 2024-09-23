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

from functools import partial
from typing import Callable, Optional
import itertools
import types
import json
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import os

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from gluonts.dataset.common import ListDataset

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.data.builder.simple import SimpleDatasetBuilder
from uni2ts.transform._base import Identity
import torch.nn.functional as F

from probts.data.lotsa_datasets import LotsaUniTSDatasetLoader
from probts.model.forecaster.prob_forecaster.moirai import Moirai

BATCH_SIZE = 10
TARGET_DATASET = "ETTh1"
SAVE_PATH = "/data/Blob_EastUS/v-zhenwzhang/tsfm_datasets/lotsa_weights"
EMBED_DIM = 0

moirai = Moirai(
    target_dim=1,
    context_length=512,
    prediction_length=96,
    freq="H",
    lags_list=[],
    patch_size="64",
    variate_mode="S",
    model_size="small",
)
moirai.moirai = moirai.moirai.cuda()


def get_dataset_embeddings(dataset: Dataset) -> np.ndarray:
    global moirai

    length = len(dataset)
    embeddings = []
    for i in tqdm(range(length), desc=f"Getting embeddings from {dataset.indexer.dataset_name}"):
        data = dataset[i]

        start_time = pd.Timestamp(data["start"].item())
        target = data["target"][0] if isinstance(data["target"], list) else data["target"]
        freq = data["freq"][0] if isinstance(data["freq"], list) else data["freq"]
        list_dataset = ListDataset(
            [{
                "start": start_time, 
                "target": target,
                "freq": freq,
            }],
            freq=freq)
        try:
            dataloader = DataLoader(
                LotsaUniTSDatasetLoader(
                    dataset="",
                    path="",
                    context_length=512,
                    history_length=512,
                    prediction_length=96,
                    scaler="none",
                    dataset_raw=list_dataset,
                ).get_iter_dataset("train"),
                batch_size=BATCH_SIZE,
            )
            batch_data = next(iter(dataloader))
            # move to cuda
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.cuda()
            _, emb = moirai.embedding(types.SimpleNamespace(**batch_data))
            emb = emb.detach()
            embeddings.append(emb.mean(axis=0).reshape(1, -1))
        except Exception as e:
            global EMBED_DIM
            assert EMBED_DIM != 0
            embeddings.append(torch.zeros(1, EMBED_DIM).cuda())
    return torch.cat(embeddings)


def get_dataset_level_weights(dataset: Dataset, target_embeddings: np.ndarray) -> np.ndarray:
    dataset_embeddings = get_dataset_embeddings(dataset)

    dataset_embeddings_norm = F.normalize(dataset_embeddings, p=2, dim=1)
    target_embeddings_norm = F.normalize(target_embeddings, p=2, dim=1)
    distances = 1 - torch.matmul(dataset_embeddings_norm, target_embeddings_norm.T)
    distances = distances.cpu().numpy()

    global SAVE_PATH, TARGET_DATASET
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    np.save(os.path.join(SAVE_PATH, f"{dataset.indexer.dataset_name}_to_{TARGET_DATASET}_distances.npy"), distances)
    return distances


def get_all_weights(train_dataset: Dataset, target_dataset: Dataset) -> dict:
    weights = {}

    target_embeddings = get_dataset_embeddings(target_dataset)
    global EMBED_DIM
    EMBED_DIM = target_embeddings.shape[1]

    if isinstance(train_dataset, ConcatDataset):
        for dataset in train_dataset.datasets:
            dataset_name = dataset.indexer.dataset_name
            weight = get_dataset_level_weights(dataset, target_embeddings)
            weights[dataset_name] = float(np.mean(weight))
    
    return weights


@hydra.main(version_base="1.3", config_name="default.yaml")
def main(cfg: DictConfig):
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    if cfg.compile:
        model.module.compile(mode=cfg.compile)
    train_dataset: Dataset = instantiate(cfg.data).load_dataset(
        defaultdict(lambda: Identity)
    )
    train_dataset_name = cfg.data._target_.split('.')[-1]

    global TARGET_DATASET
    target_dataset: Dataset = SimpleDatasetBuilder(TARGET_DATASET).load_dataset(
        defaultdict(lambda: Identity)
    )
    target_dataset.indexer.dataset_name = TARGET_DATASET

    weights = get_all_weights(train_dataset, target_dataset)

    global SAVE_PATH
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    with open(os.path.join(SAVE_PATH, f"{train_dataset_name}_to_{TARGET_DATASET}_weights.json"), "w") as f:
        json.dump(weights, f)


if __name__ == "__main__":
    main()
