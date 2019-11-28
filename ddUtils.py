import numpy as np
import mne

import sys
from typing import Iterable, Sequence, Generator, Dict

from functional import *
from events import *
from htil_math import transpose

commonEvents = {
        "problem": {
            "code": 1,
            "delay": 6000,
        },
        "strategyPrompt": {
            "code": 2,
            "delay": 13000
        }
}

EEG_SCALE = 1e-6 # Microvolts to VOLTS

TIME_SCALE = 1e-3 # Milliseconds to Seconds


def practiceEpochs(entries: Sequence[object], epochSeconds):
    return _contigEpochs(entries, epochSeconds, commonEvents)

def _contigEpochs(entries: Sequence[object], epochSeconds: int, eventDict: Dict[str,int]):
    info = _info(entries)

    data = _epochedEEG(entries, info, epochSeconds)
    timeEpochs = list(nestedMap(lambda packet: packet["timestamp"], data))

    events = matchEvents(timeEpochs, entries, eventDict)
    data = list(nestedMap(lambda sample: sample["data"], data))
    data = np.array(data)
    data = data.transpose([0,2,1])
    data *= EEG_SCALE
    events = np.concatenate( [events[:5], events[15:]], axis=0 )
    epochs = mne.EpochsArray(data, info, events=events)
    return epochs

def _info(entries):
    ind = findNext(lambda ent: ent["type"] == "eeg", entries, 0)
    eegPacket = entries[ind]
    rawInfo = eegPacket["eeg"][0]["info"]
    samplingRate = rawInfo["samplingRate"]
    channels = rawInfo["channelNames"]
    numChannels = len(channels)
    info = mne.create_info(channels,samplingRate,ch_types=["eeg"]*numChannels)
    return info


def _epochedEEG(entries, info: mne.Info, epochSeconds: int):
    # Signal data
    eeg = extract(lambda entry: entry["type"] == "eeg", entries)
    eeg = flattenSignal(eeg, "eeg")
    eeg = list(eeg)

    # Starts and ends of epochs
    epochBounds = extract(lambda ent: ent["type"] in {"fixationPoint", "end"}, entries)
    epochBounds = map(lambda ent: ent["timestamp"], epochBounds)
    epochBounds = list(epochBounds)
    numEpochs = len(epochBounds) - 1
    [starts, ends] = labeledBounds(epochBounds)
    epochSamples = int(epochSeconds * info["sfreq"]);

    data = fitEpochs(eeg, starts, ends, epochSamples)
    eeg = None #Free by dereferencing

    return data

def labeledBounds(bounds):
    def starts():
        for i in range(0, len(bounds) - 1):
            yield bounds[i]
    def ends():
        for j in range(1,len(bounds)):
            yield bounds[j]
    return (starts(),ends())

def estimateEpochLength(bounds):
    n = min(50, len(bounds) - 1)
    avg = 0.
    for i in range(n):
        avg += (bounds[i+1] - bounds[i]) / n
    return avg

def fitEpochs(eeg, starts: Iterable[float], ends: Iterable[float], size: int):
    j = 0
    data = []
    for (start, end) in zip(starts,ends):
        j = findNext(lambda samp: samp["timestamp"] >= start, eeg, j)
        if j < 0: break
        epoch = []
        sample = eeg[j]
        while len(epoch) < size and sample["timestamp"] <= end:
            epoch.append(sample)
            j += 1
            if j >= len(eeg): break
            sample = eeg[j]
        missing = size - len(epoch)
        if missing: epoch.extend( [epoch[-1] for _ in range(missing)] ) #Pad with the last value
        data.append(epoch)
    return data

