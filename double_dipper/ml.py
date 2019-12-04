# External
import mne
import numpy as np
import double_dipper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support

# Python Library
import sys


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

def lda_scorer(X, Y, testX, testY):
    return _model_score(LinearDiscriminantAnalysis(), X, Y, testX, testY)
def _model_score(model, X, Y, testX, testY):
    model.fit(X, Y)
    preds = model.predict(testX)
    (precision, recall, _, _) = precision_recall_fscore_support(testY, preds, average="binary")
    return (precision, recall)
