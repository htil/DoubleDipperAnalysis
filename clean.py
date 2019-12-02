#!/usr/bin/python3
"""
A script for putting the raw JSON data into a cleaner format

Usage: python clean.py data.json outDirectory/
"""

import pdb

# Python API
import sys
import os
import json

# External
import numpy as np

# Local
from functional import findNext, nestedMap, extract
from signal import flattenSignal
from labellers import *

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

dataPath = sys.argv[1]
outDir = sys.argv[2]

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

epochSeconds = 22 #TODO: Estimate from data automatically
numChannels = 4 #TODO: Obtain from data directly
sampleFrequency = 256 #TODO: Obtain from data directly
epochSamples = epochSeconds * sampleFrequency




def processSess(sess, condition):
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
        epoch = np.array(epoch)   # [time, channel]
        epoch = epoch.transpose() # [channel, time]
        np.save(outData, epoch)
        
        # LABELS
        meta = {"condition": condition, "path": os.path.basename(outData), "timestamp": startTime}
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

processSess(sess1, firstCond)
processSess(sess2, secondCond)
