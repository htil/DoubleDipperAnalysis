#!/usr/bin/python3
"""
A script for putting the raw JSON data into a cleaner format

Usage: python clean.py data.json outDirectory/
"""

# Python API
import sys
import os
import json

# Local
from functional import findNext, extract
from signal import flattenSignal

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

dataPath = sys.argv[1]
outPath = sys.argv[2]

assert os.path.exists(dataPath)
if not os.path.exists(outPath): os.makedirs(outPath)

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



def problemLabeller(entries):
    prob = list(filter(lambda ent: ent["type"] == "problem", entries))
    assert len(prob) == 1
    prob = prob[0]

    text = prob["problem"]
    text = "".join(text.split())
    assert "+" in text or "-" in text
    if "+" in text:
        op = "addition"
        delim = "+"
    else:
        op = "subtraction"
        delim = "-"
    [leftOperand,rightOperand] = text.split(delim)
    leftOperand = int(leftOperand)
    rightOperand = int(rightOperand)

    return {"problem": text, "left": leftOperand,"right": rightOperand,"op": op}

def correctLabeller(entries):
    isArith = lambda k: "+" in k or "-" in k
    def pred(ent):
        if ent["type"] != "submission": return False
        return any(map(isArith, ent.keys()))
    submissions = list(filter(pred, entries))
    assert len(submissions) == 1

    submission = submissions[0]
    probText = list(filter(isArith, submission.keys()))[0]
    result = submission[probText]
    if result == "correct":
        correct = True
    elif result == "incorrect":
        correct = False
    else:
        correct = None
    return {"correct": correct}

def strategyLabeller(entries):
    subs = filter(lambda ent: ent["type"] == "submission" and "strategy" in ent.keys(),entries)
    subs = list(subs)
    assert len(subs) == 1
    sub = sub[0]
    return {"strategy": sub["strategy"]}

def brightnessLabeller(entries):
    entries = filter(lambda ent: ent["type"] == "brightness", entries)
    entries = filter(lambda ent: ent["brightness"] < 1, entries)
    entries = list(entries)
    darkening = len(entries) > 0
    return {"darkening": darkening}

def toneLabeller(entries):
    entries = filter(lambda ent: ent["type"] == "tone", entries)
    entries = list(entries)
    tone = len(entries) > 0
    return {"tone": tone}

def processSess(sess, condition):
    commonMeta = {"condition": condition}
    labellers = []


    def findBounds(ind):
        nextPoint = lambda ent: ent["type"] == "fixationPoint"
        ind = findNext(nextPoint, sess, ind)
        if ind == -1: return (-1, -1)
        start = ind

        ind = findNext(lambda ent: ent["type"] == "fixationPoint", sess, ind+1)
        if ind == -1: return (start, len(sess) - 1)

        end = ind + 1
        return (start, end)


    epochNo = 0
    i = 0
    (start, end) = findBounds(0)
    epoch
    while start != -1:

        epochEnts = sess[start:end]
        eeg = extract(lambda ent: ent["type"] == "eeg", epochEnts)
        eeg = flattenSignal(eeg, "eeg")
        eeg = map(lambda ent: ent["data"], eeg)
        eeg = list(eeg)

        if len(eeg) < epochSamples:
            slack = epochSamples - len(eeg)
            eeg.extend(eeg[-1] for _ in range(slack))
        eeg = np.array(eeg)   # [time, channel]
        eeg = eeg.transpose() # [channel, time]
        
        (start,end) = findBounds(end)























