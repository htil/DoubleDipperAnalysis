import numpy as np

from functional import findNext

def matchEvents(timeEpochs, entries, eventDict):
    """
    data must have axes [epoch,sample,channel]
    """
    eventEnts = {}
    numEvents = 0
    for key in eventDict.keys():
        instances = list(filter(lambda ent: ent["type"] == key, entries))
        eventEnts[key] = instances
        numEvents += len(instances)

    events = np.zeros([numEvents,3], np.int64)

    i = 0
    for (eventName, ents) in eventEnts.items():
        code = eventDict[eventName]["code"]
        
        if True:
            delay = eventDict[eventName]["delay"]
            subArray = _matchWithDelay(timeEpochs, ents, code, delay)
        else:
            pass

        events[i:i+len(subArray),:] = subArray  
        i += len(subArray)


    return events

def _matchWithDelay(timeEpochs, ents, code, delay):
    assert len(ents) == len(timeEpochs)
    epochSize = len(timeEpochs[0])

    events = np.zeros([len(ents), 3], np.int64)
    events[:,2] = code

    for (i,epoch) in enumerate(timeEpochs):
        epochStart = epoch[0]
        eTime = delay + epochStart
        j = findNext(lambda time: time >= eTime, epoch, 0)
        events[i,0] = (epochSize*i) + (j - 1)
    return events

def _matchWithTimestamp(timeEpochs, ents, code):
    events = np.zeros([len(ents),3], np.int64)
    events[:,2] = code

    packetTimes = [t for e in timeEpochs for t in e]
    j = 0
    for ent in ents:
        eTime = ent["timestamp"]
        j = findNext(lambda time: time >= eTime, packetTimes, j)
        events[ind,0] = j - 1
        events[ind,2] = eventDict[eventName]["code"]
        ind += 1


