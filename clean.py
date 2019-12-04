#!/usr/bin/python3
"""
A script for putting the raw JSON data into a cleaner format

Usage: python clean.py subjNumber data.json outDirectory/
"""

import pdb

# Python API
import sys
import os
import json

# External
import numpy as np

# Local
from double_dipper import constants
from double_dipper.functional import findNext, nestedMap, extract
from double_dipper.signal import flattenSignal
from double_dipper.labellers import *

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

subjNumber = int(sys.argv[1])
dataPath = sys.argv[2]
outDir = sys.argv[3]

assert os.path.exists(dataPath)
if not os.path.exists(outDir): os.makedirs(outDir)

with open(dataPath, "r") as r: data = json.load(r)

ind = findNext(lambda ent: ent["type"] == "condition", data, 0)
firstCond = data[ind]["condition"]
ind = findNext(lambda ent: ent["type"] == "condition", data, ind + 1)
secondCond = data[ind]["condition"]

sess1 = data[:ind]
sess2 = data[ind:]
data = None

EEG_SCALE = constants.eeg_scale
epochSeconds = constants.epoch_length
numChannels = len(constants.channel_names)
sampleFrequency = constants.sfreq
epochSamples = epochSeconds * sampleFrequency



epochNo = 0
def processSess(sess, condition):
    global epochNo
    labelFuncs = [correctLabeller, strategyLabeller, brightnessLabeller, toneLabeller]

    eeg = extract(lambda ent: ent["type"] == "eeg", sess)
    eeg = flattenSignal(eeg, "eeg")
    eeg = nestedMap(lambda ent: ent, eeg)
    eeg = list(eeg)

    # Forgot to add timestamps to problems, so they have to be taken care of separately
    problems = list(extract(lambda ent: ent["type"] == "problem", sess))


    def findBounds(ind):
        nextPoint = lambda ent: ent["type"] == "fixationPoint"
        ind = findNext(nextPoint, sess, ind)
        if ind == -1: return (-1, -1)
        start = ind
        ind = findNext(lambda ent: ent["type"] == "fixationPoint", sess, ind+1)
        if ind == -1: return (start, len(sess))
        end = ind + 1
        return (start, end)

    i = 0
    (start, end) = findBounds(0)
    while start != -1:
        outPath = f"{condition}_epoch{i}"
        outData = os.path.join(outDir, outPath + ".npy")
        outMeta = os.path.join(outDir, outPath + "_labels" + ".json")
        startTime = sess[start]["timestamp"]
        endTime = sess[end-1]["timestamp"]

        epoch = filter(lambda ent: startTime <= ent["timestamp"] and ent["timestamp"] <= endTime, eeg)
        epoch = map(lambda ent: ent["data"], epoch)
        epoch = list(epoch)

        if len(epoch) < epochSamples:
            slack = epochSamples - len(epoch)
            epoch.extend(epoch[-1] for _ in range(slack))
        epoch = epoch[:epochSamples] # Trim if it's too long
        epoch = np.array(epoch)   # [time, channel]
        epoch = epoch.transpose() # [channel, time]
        epoch *= EEG_SCALE
        np.save(outData, epoch)
        
        # LABELS
        meta = {"condition": condition,
                "path": os.path.basename(outData),
                "timestamp": startTime,
                "epoch": epochNo,
                "id": subjNumber
        }
        problem = problems[i]
        meta.update(getProblem(problem))
        def pred(ent):
            if "timestamp" not in ent: return True
            return startTime <= ent["timestamp"] and ent["timestamp"] <= endTime
        evs = list(filter(pred, sess))
        for f in labelFuncs:
            meta.update(f(evs))
        with open(outMeta, "w") as w: json.dump(meta,w,indent=2)

        (start,end) = findBounds(end-1)
        i += 1
        epochNo += 1

processSess(sess1, firstCond)
processSess(sess2, secondCond)
