import numpy as _np

"""
Sample frequency in Hz
"""
sfreq = 256

"""
Factor that all EEG readings should be multiplied by before being used in mne
"""
eeg_scale = 1e-6

channel_names = ["EEG1", "EEG2", "EEG3", "EEG4"]

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
