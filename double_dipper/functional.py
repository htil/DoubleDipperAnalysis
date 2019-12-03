from typing import Iterable, Sequence, Generator

def nestedMap(fn, sequences):
    for seq in sequences:
        yield from map(fn, seq)

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
