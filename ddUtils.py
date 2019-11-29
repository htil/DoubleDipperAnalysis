import numpy as np
import mne

import sys
from typing import Iterable, Sequence, Generator, Dict

from functional import *
import events
from events import *
from htil_math import transpose

commonEvents = {
        "problem": {"code": 2, "delay": 6000},
        "strategyPrompt": {"code": 3, "delay": 13000}
}

EEG_SCALE = 1e-6 # Microvolts to VOLTS

TIME_SCALE = 1e-3 # Milliseconds to Seconds

EPOCH_SECONDS = 22


def practiceEpochs(entries: Sequence[object], alsoEvents=False):
    return _contigEpochs(entries, EPOCH_SECONDS, commonEvents, alsoEvents=alsoEvents)

def practiceRaw(entries: Sequence[object], average=False):
    startInd = findNext(lambda ent: ent["type"] == "fixationPoint", entries, 0)
    startTime = entries[startInd]["timestamp"]
    raw = _raw(entries)

    if average:
        data = raw.get_data()
        epochSamples = int(EPOCH_SECONDS * raw.info["sfreq"])
        rem = data.shape[1] % epochSamples
        if rem: data = data[:, :-rem]
        data = data.reshape( [data.shape[0], -1, epochSamples] ) #[channel, epoch, sample]
        data = data.transpose([1,0,2])                           #[epoch, channel, sample]
        data = np.mean(data, axis=0) #[channel, sample]
        raw = mne.io.RawArray(data, raw.info)

        evs = np.zeros([len(commonEvents),3], np.int64)
        for (i, (evName, meta)) in enumerate(commonEvents.items()):
            code = meta["code"]
            delaySeconds = meta["delay"] / 1000
            sampleNo = delaySeconds * raw.info["sfreq"]
            evs[i,0] = sampleNo
            evs[i,2] = code
    else:
        evs = events.match2Raw(raw, entries, commonEvents)

    return [raw,evs]

def _raw(entries: Sequence[object]):
    info = _info(entries)
    # Signal data
    eeg = extract(lambda entry: entry["type"] == "eeg", entries)
    eeg = flattenSignal(eeg, "eeg")
    eeg = map(lambda ent: ent["data"], eeg)
    eeg = list(eeg)
    eeg = np.array(eeg)
    eeg = eeg.transpose()
    eeg *= EEG_SCALE
    raw = mne.io.RawArray(eeg, info)
    return raw

def _contigEpochs(entries: Sequence[object], epochSeconds: int, eventDict: Dict[str,int], alsoEvents=False):


    data = _epochedEEG(entries, info, epochSeconds)
    timeEpochs = list(nestedMap(lambda packet: packet["timestamp"], data))

    events = matchEvents(timeEpochs, entries, eventDict)
    data = list(nestedMap(lambda sample: sample["data"], data))
    data = np.array(data)
    data = data.transpose([0,2,1])
    data *= EEG_SCALE
    epochs = mne.EpochsArray(data, info, events=None) #TODO: Find out how to have more than one event per epoch

    if alsoEvents: return [epochs, events]
    return epochs



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

def _info(entries):
    ind = findNext(lambda ent: ent["type"] == "eeg", entries, 0)
    eegPacket = entries[ind]
    rawInfo = eegPacket["eeg"][0]["info"]
    samplingRate = rawInfo["samplingRate"]
    channels = rawInfo["channelNames"]
    numChannels = len(channels)
    info = mne.create_info(channels,samplingRate,ch_types=["eeg"]*numChannels)
    return info

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

