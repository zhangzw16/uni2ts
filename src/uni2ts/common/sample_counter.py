import json
import pickle
from pathlib import Path

import numpy as np
from typing import List, Union
from collections import defaultdict

def initialize_counter_global():
    global sample_counter
    sample_counter = SampleCounter()

class SampleCounter:
    def __init__(self):
        self.dataset_length = defaultdict(int)
        self.counter = {}
        self.intervals = {}

        # some temporary variables
        self.cur_dataset = None
        self.tmp_record = []
        self.var_shape = None
        self.rand_idxs = None

    def initialize_counter(self, dataset, variable_name, shape):
        if variable_name not in self.counter[dataset]:
            self.counter[dataset][variable_name] = np.zeros(shape, dtype=int)
            self.intervals[dataset][variable_name] = []

    def record_candidate_variables(
            self, 
            cur_dataset: str, 
            candidate_variables: Union[List[str], str], 
            shape: tuple
        ):
        self.cur_dataset = cur_dataset
        if cur_dataset not in self.counter:
            self.counter[cur_dataset] = {}
            self.intervals[cur_dataset] = {}

        if isinstance(candidate_variables, str):
            candidate_variables = [candidate_variables]
        candidate_variables = list(candidate_variables)
        for var in candidate_variables:
            self.initialize_counter(cur_dataset, var, shape)
        if len(shape) == 2:
            candidate_variables = [var for var in candidate_variables for _ in range(shape[0])]
        self.var_shape = shape
        self.tmp_record = candidate_variables

    def update_selected_vars(self, rand_idxs):
        self.rand_idxs = rand_idxs
        self.tmp_record = [self.tmp_record[idx] for idx in rand_idxs]

    def update_crop(self, a, b):
        # add intervals to the dict
        self.intervals[self.cur_dataset][self.tmp_record[0]].append((a, b))

        for var, idx in zip(self.tmp_record, self.rand_idxs):
            if self.counter[self.cur_dataset][var] is None:
                raise ValueError(f"Variable {var} not initialized")
            if len(self.var_shape) == 2:
                n_vars = self.var_shape[0]
                assert self.counter[self.cur_dataset][var].ndim == 2
                self.counter[self.cur_dataset][var][idx%n_vars, a:b] += 1
            else:
                self.counter[self.cur_dataset][var][a:b] += 1

    def get_counter(self, dataset, variable_name):
        return self.counter[dataset][variable_name]

    def to_dict(self):
        # calculate the maximum sample count for each variable
        max_counts = defaultdict(lambda: defaultdict(lambda: None))
        for dataset in self.counter:
            for var in self.counter[dataset]:
                if self.counter[dataset][var] is not None:
                    max_counts[dataset][var] = int(np.max(self.counter[dataset][var]))
        return max_counts

    def save_to_files(self, path=Path("./debug")):
        print(f"Saving ts_sample_counts to json file consists of {len(self.counter)} datasets...")
        # save dict to json file
        with open(path / "ts_sample_counts.json", "w") as f:
            json.dump(self.to_dict(), f)
        # save dict to pickle file
        with open(path / "ts_sample_counts.pkl", "wb") as f:
            pickle.dump(self.counter, f)
        with open(path / "ts_sample_intervals.pkl", "wb") as f:
            pickle.dump(self.intervals, f)
