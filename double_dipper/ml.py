# External
import mne
import numpy as np
import double_dipper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Python Library
import sys
from itertools import product

def grid_search(X, Y, tsX, tsY, feature_selectors=None, resamplers=None, models=None, verbose=True):
    if not models:
        models = [LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis]
    if not feature_selectors: feature_selectors = [None]
    if not resamplers: resamplers = [None]

    arg_names = ["model", "resampler", "feature_selector"]
    axes = [models, resamplers, feature_selectors]

    inds = np.zeros(len(axes), np.int8)
    best_inds = np.zeros(len(axes), np.int8)
    best_conf = np.zeros([2, 2]) 
    best_f1 = -float("inf")

    for inds in product(*[range(len(axis)) for axis in axes]):
        vals = [axis[i] for (i, axis) in zip(inds, axes)]
        kwargs = {name:val for (name, val) in zip(arg_names, vals)}
        conf = _test_instance(X, Y, tsX, tsY, **kwargs)
        (prec, rec, f1) = _extract_metrics(conf)
        if verbose:
            out_str = ",".join(f"{name}={ind}" for (name, ind) in zip(arg_names, inds))
            out_metrics = "precision={:.3f}, recall={:.3f}, f1={:.3f}".format(prec, rec, f1)
            print(out_str, ":", sep="", end=" ")
            print("\t", out_metrics, sep="")
        if f1 > best_f1:
            best_inds[:] = inds[:]
            best_conf = conf
            best_f1 = f1
            if verbose: print("New best achieved")
        if verbose: print()

    return (best_inds, best_conf)


def _test_instance(X, Y, tsX, tsY, feature_selector, resampler, model):
    if feature_selector:
        X = feature_selector(X)
        tsX = feature_selector(tsX)
    if resampler:
        n_lacking = len(Y) // 2 - np.sum(Y)
        if n_lacking: (X, Y) = resampler().fit_resample(X, Y)

    mod = model()
    mod.fit(X, Y)
    preds = mod.predict(tsX)
    return confusion_matrix(tsY, preds)

def _extract_metrics(conf):
    prec = conf[1, 1] / np.sum(conf[:, 1])
    rec = conf[1, 1] / np.sum(conf[1, :])
    f1 = 2*prec*rec / (prec + rec)
    return (prec, rec, f1)



def temporal_cross_validation(X, Y, resampler = None, model = None, splits=[.2,.4,.6,.8], verbose = True):
    if not model: model = LinearDiscriminantAnalysis
    precision = np.zeros(len(splits))
    recall = np.zeros(len(splits))
    for (i, spl) in enumerate(splits):
        split_ind = int(len(X) * spl)
        (trX, trY) = (X[:split_ind], Y[:split_ind])
        if resampler and not _too_skewed(trY):
            (trX,trY) = resampler().fit_resample(trX, trY)

        mod = model()
        mod.fit(trX, trY)
        preds = mod.predict(X[split_ind:])
        (precision[i], recall[i], _, _) = precision_recall_fscore_support(Y[split_ind:], preds, average="binary")
    return (precision, recall)

def _too_skewed(Y, k_neighbors=5):
    n_neighbors = k_neighbors + 1
    numPos = np.sum(Y)
    numNeg = np.prod(Y.shape) - numPos
    return (numPos < n_neighbors) or (numNeg >= n_neighbors)

def cross_validation(dataset, scorer = None, verbose = True):
    if not scorer: scorer = lda_scorer
    keys = list(dataset.keys())
    numFolds = len(keys)
    precision = np.zeros(numFolds)
    recall = np.zeros(numFolds)
    for (i, testKey) in enumerate(keys):
        testX = dataset[testKey]["x"]
        testY = dataset[testKey]["y"]
        others = [dataset[key] for key in keys if key != testKey]
        X = np.concatenate( list(map(lambda part: part["x"], others)) , axis = 0 )
        Y = np.concatenate( list(map(lambda part: part["y"], others)), axis = 0)
        (precision[i], recall[i]) = scorer(X, Y, testX, testY)
        if verbose:
            sys.stderr.write(f"Held out fold {str(testKey)}: precision = {precision[i]}, recall = {recall[i]}\n")
    return (precision, recall)

def model_predictor(X, Y, testX, model):
    model.fit(X, Y)
    return model.predict(testX)

def lda_predictor(X, Y, testX):
    model = LinearDiscriminantAnalysis()
    model.fit(X, Y)
    return model.predict(testX)

def lda_scorer(X, Y, testX, testY):
    return _model_score(LinearDiscriminantAnalysis(), X, Y, testX, testY)
def _model_score(model, X, Y, testX, testY):
    model.fit(X, Y)
    preds = model.predict(testX)
    (precision, recall, _, _) = precision_recall_fscore_support(testY, preds, average="binary")
    return (precision, recall)
