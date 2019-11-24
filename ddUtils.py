import numpy as np

from typing import Iterable, Sequence, Generator

def contigEpochs(entries: Sequence[object]):
    epochBounds = extract(lambda ent: ent["type"] in {"fixationPoint", "end"}, entries)
    epochBounds = map(lambda ent: ent["timestamp"], epochBounds)
    epochBounds = list(epochBounds)

    print("epoch lenghts in time:")
    i = 0
    while i < len(epochBounds) - 1:
        print( epochBounds[i+1] - epochBounds[i] )
        i += 1

    #eeg = extract(lambda entry: entry["type"] == "eeg", entries)
    #eeg = flattenSignal(eeg, "eeg")
    #eeg = list(eeg)
    #eeg = np.array(eeg, dtype=np.float32, copy=False)

    #info = eeg[0]["info"]
    #samplingRate = info["samplingRate"]



def flattenSignal(data: Iterable[object], key) -> Generator[object,None,None]:
    if callable(key): fn = key
    else:             fn = lambda entry: entry[key]
    for entry in map(fn, data):
        yield from entry

def mapInPlace(fn, seq: Sequence[object]):
    for (i, el) in enumerate(seq):
        seq[i] = fn(el)

def extract(pred, seq: Sequence[object]) -> Iterable[object]:
    i = 0
    while i < len(seq):
        el = seq[i]
        if pred(el):
            yield el
            del seq[i]
        else:
            i += 1
