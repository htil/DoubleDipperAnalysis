import os
import glob

def filePairs(directory):
    directory = os.path.relpath(directory) # Trims tailing '/'
    patt = f"{directory}/*.npy"

    for dataPath in glob.glob(patt):
        metaPath = dataPath.split(".")[0] + "_labels.json"
        yield (dataPath, metaPath)
