# Python API
from functools import reduce
from typing import Callable

# External
import numpy as np
import mne
mne.set_log_level("CRITICAL")

# Local
from . import constants

FeatureFunction = Callable[[np.ndarray], np.ndarray]

def chain(*functions: FeatureFunction) -> FeatureFunction:
    def chained(args):
        for fn in functions:
            args = fn(args)
        return args
    return chained

def dup(*functions: FeatureFunction, add=False) -> FeatureFunction:
    def dup_func(X):
        to_cat = [fn(X) for fn in functions]
        if add: to_cat.insert(0, X)
        to_cat = [flatten_end(dat) for dat in to_cat]
        return np.concatenate(to_cat, axis=-1)
    return dup_func

def time_window(start: int, end: int) -> FeatureFunction:
    firstSample = start * constants.sfreq
    lastSample = end * constants.sfreq
    return lambda X: X[...,firstSample:lastSample]

def bandpass_filter(l_freq, h_freq) -> FeatureFunction:
    return lambda X: mne.filter.filter_data(X, constants.sfreq, l_freq=l_freq, h_freq=h_freq)

def psd(fmin=0, fmax=np.inf, tmin=None, tmax=None, psd_comp=None, add=False):
    if not psd_comp: psd_comp = mne.time_frequency.psd_welch
    def psd_func(X):
        raw = mne.EpochsArray(X, constants.info)
        (psds, freqs) = psd_comp(raw, fmin, fmax, tmin, tmax)
        return np.concatenate([X, psds], axis=-1) if add else psds
    return psd_func

def psd_bands(psd_comp=None, add=False) -> FeatureFunction:
    bands = np.array([
        [1, 3], #Delta
        [4, 7], #Theta
        [8, 11], #Alpha
        [12, 29], #Beta
        [30, 45]  #Gamma
    ])
    psd_func = psd(fmin=np.min(bands), fmax=np.max(bands), psd_comp=psd_comp)
    def band_func(X):
        psds = psd_func(X)
        band_powers = np.zeros([X.shape[0], bands.shape[0]])
        for (i, (l_freq, h_freq)) in enumerate(bands):
            band_powers[:, i] = np.sum(bands[:, l_freq-1:h_freq-1])
        X = flatten_end(X)
        return np.concatenate([flatten_end(X), band_powers], axis=-1) if add else band_powers
    return band_func

def flatten_end(X):
    return X.reshape([X.shape[0], np.prod(X.shape[1:])])
