import numpy as _np
import mne as _mne

"""
Sample frequency in Hz
"""
sfreq = 256

"""
Factor that all EEG readings should be multiplied by before being used in mne
"""
eeg_scale = 1e-6

"""
Names of the EEG channels according to 10-10 nomenclature
"""
channel_names = ["Tp9", "Fp1", "Fp2", "Tp10"]

"""
Length of epoch in seconds
"""
epoch_length = 22

"""
Number of problems/epochs in regular experiment
"""
num_problems = 112

"""
Number of problems/epochs in a practice experiment"
"""
num_practice = 10

"""
MNE info object about channels and sampling frequency
"""
info = _mne.create_info(channel_names, sfreq, ch_types=["eeg"]*len(channel_names))


class _Event(object):
    def __init__(self, code, delay):
        self.code = code
        self.delay = delay

    @property
    def sample(self):
        return int(self.delay * sfreq)

    @property
    def occurrence(self):
        return _np.array([self.sample, 0, self.code])

problem = _Event(2, 6)
strategy_prompt = _Event(3, 13)

tone = _Event(4, 8)
darkening = _Event(5, 10)
