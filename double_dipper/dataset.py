from . import constants

def mapDataset(fn_x, dataset, fn_y = None):
    for (key, fold) in dataset.items():
        fold["x"] = fn_x(fold["x"])
        if fn_y: fold["y"] = fn_y(fold["y"])
