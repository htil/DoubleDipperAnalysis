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
        if ent["type"] != "submission": return False
        return any(map(isArith, ent.keys()))
    submissions = list(filter(pred, entries))
    if len(submissions) != 1:
        print(submissions)
        raise Exception("More than one submission found for epoch")

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
    sub = subs[0]
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

