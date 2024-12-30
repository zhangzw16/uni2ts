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

import yaml
import os
import torch
import itertools

from enum import Enum
from typing import Any, Union, List

import numpy as np
from torch.utils.data import Dataset
from torch import Tensor

from uni2ts.common.env import env
from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from uni2ts.data.indexer import Indexer
from uni2ts.transform import Transformation
import uni2ts.common.sample_counter as sc

class SampleTimeSeriesType(Enum):
    """
    How to sample from the dataset.
    - none: do not sample, return the current index.
    - uniform: each time series sampled with equal probability
    - proportional: each time series sampled with probability proportional to it's length
    """

    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    # Class variable to cache the YAML data
    _yaml_data_cache = None
    # for counting how many times a time series is sampled
    # ts_sample_counts = {} # needs num_workers=0

    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        """
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight
        
        
        # Load the YAML file only once
        # if TimeSeriesDataset._yaml_data_cache is None:
        #     try:
        #         prob_path = env.LOTSA_V1_DISTANCES_FILE
        #         with open(prob_path, 'r') as f:
        #             TimeSeriesDataset._yaml_data_cache = yaml.safe_load(f)
        #         print(f"Loaded YAML data from {prob_path}")
        #     except Exception as e:
        #         print(f"Error loading YAML file: {e}")
        #         TimeSeriesDataset._yaml_data_cache = {}
        data = TimeSeriesDataset._yaml_data_cache

        try:
            if data is None:
                raise KeyError
            series_prob = data[indexer.dataset_name]
            self.weights = self._convert_to_cycle(series_prob)
            print(f"Loaded distances for {indexer.dataset_name}")
        except KeyError:
            print(f"No probability found for {indexer.dataset_name}")
            self.weights = np.ones(len(self)) * 0.42  # Adjust as needed

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

        print("Loaded dataset: ", indexer.dataset_name, " with weight: ", dataset_weight)

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        """
        Obtain a time series from the dataset, flatten
        :param idx: index of time series to retrieve. if sample_time_series is specified, this will be ignored.
        :return: transformed time series data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )
        idx_ori = idx
        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)

        # print(f"Sample from {self.indexer.dataset_name} at index {idx}, original index {idx_ori}")
        samples = self._get_data(idx)
        return self.transform(self._flatten_data(samples))
 
    @property
    def num_ts(self) -> int:
        """
        Get the number of time series in the dataset
        """
        return len(self.indexer)

    def _sample_ts(self, samples) -> None:
        dataset_name = self.indexer.dataset_name
        if isinstance(samples['target'], np.ndarray):
            shape = samples['target'].shape
        else:
            shape = (samples['target'][0].shape[0],)
        try:
            sc.sample_counter.record_candidate_variables(dataset_name, samples['item_id'], shape)
        except Exception as e:
            pass

    def __len__(self) -> int:
        """
        Length is the number of time series multiplied by dataset_weight
        """
        return int(np.ceil(self.num_ts * self.dataset_weight))

    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        """
        Obtains time series from Indexer object
        """
        samples = self.indexer[idx % self.num_ts]
        self._sample_ts(samples)
        return self.indexer[idx % self.num_ts]
    
    def _convert_to_cycle(self, series_prob: dict[str, float]) -> list[float]:
        """
        Convert series probabilities to weights
        """
        total_length = self.__len__()
        one_cycle = [series_prob[key] for key in series_prob]
        assert len(one_cycle) == self.num_ts, f"Number of series probabilities {len(one_cycle)} must match number of time series {self.num_ts}"
        return (one_cycle * (total_length // len(one_cycle) + 1))[:total_length]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        """
        Convert time series type data into a list of univariate time series
        """
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    """
    Samples multiple time series and stacks them into a single time series.
    Underlying dataset should have aligned time series, meaning same start and end dates.
    """

    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param max_ts: maximum number of time series that can be stacked together
        :param combine_fields: fields which should be stacked
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        :param sampler: how to sample the other time series
        """
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        # relevant_weights = np.concatenate([self.weights[:idx], self.weights[idx+1:]])
        # probabilities = softmin(relevant_weights)
        # others = np.random.choice(choices, n_series - 1, replace=False, p=probabilities)
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        self._sample_ts(samples)
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class EvalDataset(TimeSeriesDataset):
    """
    Dataset class for validation.
    Should be used in conjunction with Eval transformations.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
    ):
        """
        :param windows: number of windows to perform evaluation on
        """
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows, # repeat the dataset `windows` times
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item
