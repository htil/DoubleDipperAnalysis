# External
import numpy as np

# Python API
import os
import glob
import json
from collections import defaultdict
from typing import Tuple, Iterable

def filePairs(*directories):
    for directory in directories:
        directory = os.path.relpath(directory) # Trims tailing '/'
        patt = f"{directory}/*.npy"
        for dataPath in glob.glob(patt):
            metaPath = dataPath.split(".")[0] + "_labels.json"
            yield (dataPath, metaPath)


def partition(divider, labeller, pairs: Iterable[Tuple[str,str]]):
    parts = defaultdict(lambda: {"x": [], "y": []})
    for (dataPath, metaPath) in pairs:
        with open(metaPath, "r") as r: metadata = json.load(r)

        part = divider(metadata)
        if partition is None: continue
        label = labeller(metadata)
        if label is None: continue

        data = np.load(dataPath)
        parts[part]["x"].append(data)
        parts[part]["y"].append(label)

    for (_, d) in parts.items():
        data = np.stack(d["x"])
        labels = np.stack(d["y"])
        d["x"] = data
        d["y"] = labels
    return parts

