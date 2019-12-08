# External
import mne
import numpy as np
import double_dipper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support

# Python Library
import sys

def temporal_cross_validation(X, Y, scorer = None, splits=[.2,.4,.6,.8], verbose = True):
    if not scorer: scorer = lda_scorer
    precision = np.zeros(len(splits))
    recall = np.zeros(len(splits))
    for (i, spl) in enumerate(splits):
        split_ind = int(len(X) * spl)
        (precision[i], recall[i]) = scorer(X[:split_ind], Y[:split_ind], X[split_ind:], Y[split_ind:])
    return (precision, recall)

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
