import pdb
def getProblem(entry):
    prob = entry
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
        if ent["type"] == "skipForm": return True
        if ent["type"] != "submission": return False
        return any(map(isArith, ent.keys()))
    submissions = list(filter(pred, entries))

    # The "skipForm" in pred could have captured a strategy submission,
    # so choose 0 to always get the problem answer submission
    submission = submissions[0]

    if submission["type"] == "skipForm":
        return {"answer": "skipped"}

    probText = list(filter(isArith, submission.keys()))[0]
    result = submission[probText]
    assert result in {"correct", "incorrect"}
    answer = "correct" if result == "correct" else "incorrect"
    return {"answer": answer}

def strategyLabeller(entries):
    def pred(ent):
        if ent["type"] == "skipForm": return True
        return ent["type"] == "submission" and "strategy" in ent.keys()
    subs = filter(pred,entries)
    subs = list(subs)

    # Fat-fingered the `Submit` button and lost the data
    if not subs: return {"strategy": None}

    # The "skipForm" pred could have captured the problem answer submission, so do -1 to get last submission
    sub = subs[-1]
    if sub["type"] == "skipForm": return {"strategy": None}
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

