from . import constants

import numpy as np
from collections import defaultdict

def mapDataset(fn_x, dataset, fn_y = None):
    for (key, fold) in dataset.items():
        fold["x"] = fn_x(fold["x"])
        if fn_y: fold["y"] = fn_y(fold["y"])

map_dataset = mapDataset

def subset(pred, dataset):
    subset = defaultdict(lambda: {"x":[], "y":[]})
    for (key, fold) in dataset.items():
        if pred(key): subset[key] = fold
    return subset

def avg_data(dset) -> np.ndarray:
    return np.mean(
        np.concatenate([dset[k]["x"] for k in dset.keys()], axis=0)
        , axis=0
    )
