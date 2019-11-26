import numpy as np
import mne

import sys
from functools import reduce
from typing import Iterable, Sequence, Generator

def contigEpochs(entries: Sequence[object], epochSeconds: int = None, verbose=True):
    eeg = extract(lambda entry: entry["type"] == "eeg", entries)
    eeg = flattenSignal(eeg, "eeg")
    eeg = list(eeg)

    rawInfo = eeg[0]["info"]
    samplingRate = rawInfo["samplingRate"]
    channels = rawInfo["channelNames"]
    numChannels = len(channels)

    info = mne.create_info(channels,samplingRate,ch_types=["eeg"]*numChannels)


    epochBounds = extract(lambda ent: ent["type"] in {"fixationPoint", "end"}, entries)
    epochBounds = map(lambda ent: ent["timestamp"], epochBounds)
    epochBounds = list(epochBounds)
    numEpochs = len(epochBounds) - 1


    if not epochSeconds:
        n = min(50, len(epochBounds) - 1)
        avg = 0.
        for i in range(n):
            avg += (epochBounds[i+1] - epochBounds[i]) / n
        epochSeconds = int(avg / 1e3)
    epochSamples = epochSeconds * samplingRate;


    if verbose: sys.stderr.write(f"Obtaining {numEpochs} epochs\n");
    data = []
    j = 0
    for i in range(numEpochs):
        [start,end] = epochBounds[i:i+2]
        j = findNext(lambda samp: samp["timestamp"] >= start, eeg, j)
        if j < 0: break

        count = 0
        sample = eeg[j]
        while count < epochSamples and sample["timestamp"] <= end:
            data.append(sample["data"])
            count += 1
            j += 1
            if j >= len(eeg): break
            sample = eeg[j]
        missing = epochSamples - count
        if missing: data.extend( [data[-1] for _ in range(missing)] ) #Pad with the last value
    eeg = None #Free by dereferencing

    data = np.array(data)
    data /= 1e6 # Convert from microvolts to volts
    data = np.reshape(data, [numEpochs, numChannels, epochSamples])


    epochs = mne.EpochsArray(data, info)
    return epochs



def findNext(pred, seq, start=0):
    for i in range(start, len(seq)):
        if pred(seq[i]): return i
    return -1

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
