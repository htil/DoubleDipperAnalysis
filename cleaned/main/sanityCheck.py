#!/usr/bin/python3
import json
import os
import sys
import glob

def check(partNo):
    counts = {}
    for cond in ["C1", "C2"]:
        darkCount = 0
        toneCount = 0
        paths = glob.glob(f"{partNo}/{cond}_epoch*_labels.json")
        for path in paths:
            with open(path, "r") as r: data = json.load(r)
            darkCount += data["darkening"]
            toneCount += data["tone"]
        if darkCount != len(paths) // 2:
            raise Exception(f"{partNo}: Wrong number of darkenings")
        if cond == "C1" and toneCount != len(paths):
            raise Exception(f"{partNo}: Wrong number of tones for {cond}")
        counts[cond] = {"darkening": darkCount, "paths": len(paths)}

    for key in counts["C1"].keys():
        assert counts["C1"][key] == counts["C2"][key]
    sys.stderr.write(f"Checked participant {partNo}\n")

if len(sys.argv) > 1:
    partNo = int(sys.argv[1])
    check(partNo)
else:
    for path in glob.glob("*"):
        if path.endswith(".py"): continue
        partNo = int(path)
        check(partNo)
