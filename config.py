# -*- coding: utf-8 -*-
"""Global configuration: data paths, sampling rate, splits and model hyper-parameters.

Data path can be set with the environment variable RVAD_DATA_DIR, otherwise the
default ``./data`` (relative to this file) is used. See README.md for the
expected directory layout produced by ``bids_to_npz.py``.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------- data ----------
DATA_DIR = Path(os.environ.get('RVAD_DATA_DIR', HERE / 'data'))
DATA_SUBDIR = 'pp_eeg_v2'          # sub-folder holding the epoched .npz files
SFREQ = 1000                       # Hz; epochs are 1001 samples (-200 to 800 ms)
AVG_CHANNELS = True                # True: average the 64 channels into one waveform (N, 1, T)
                                   # False: keep all channels (N, 64, T)

# ---------- task filter ----------
TASK_FILTER = 'TC_AJ'              # keypress task only (TC_MS = mental counting)
                                   # None keeps every task

# ---------- per-participant split ----------
SPLIT = (0.8, 0.1, 0.1)            # 80 % train / 10 % validation (early stopping) / 10 % test
RANDOM_STATE = 42

# ---------- deep-learning training ----------
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8                       # early-stopping patience (validation balanced accuracy)
NUM_WORKERS = 0                    # keep 0 on Windows

# ---------- SVM (on the flattened ERP waveform) ----------
SVM = dict(
    svm_C=1.0,
    kernel='rbf',
)

# ---------- EEGNet ----------
EEGNET = dict(
    F1=8,
    D=2,
    kernel_1=64,                   # length at 128 Hz; rescaled by sfreq / 128
    kernel_2=16,
    drop_rate=0.25,
)

# ---------- DeepConvNet ----------
DEEPCONVNET = dict(
    F=25,
    conv_temporal=10,
    drop_rate=0.5,
)

# ---------- FBCNet ----------
FBCNET = dict(
    band_edges=[4, 8, 12, 16, 20],
    spatial_filters=32,
    num_windows=8,
    drop_rate=0.3,
)

# ---------- output ----------
RESULTS_DIR = Path(os.environ.get('RVAD_RESULTS_DIR', HERE / 'results'))
