def flattenSignal(data, key):
    if callable(key): fn = key
    else:             fn = lambda ent: ent[key]
    return map(fn, data)
