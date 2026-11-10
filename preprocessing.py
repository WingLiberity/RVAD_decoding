# -*- coding: utf-8 -*-
"""预处理：逐通道 z-score 标准化、带通滤波（FBCNet 频带组）。"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def zscore_fit(X):
    """基于训练集计算逐通道均值/标准差。X: (N, C, T)。"""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(X, mean, std):
    """按训练集统计量标准化。"""
    return ((X - mean) / std).astype(np.float32)


def bandpass(X, f1, f2, sfreq, order=4):
    """零相位带通滤波（Butterworth，sosfiltfilt）。X: (..., T)，沿最后一维滤波。"""
    sos = butter(order, [f1, f2], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1)


def filter_bands(X, band_edges, sfreq, order=4, chunk=1024):
    """对每个频带逐块滤波，返回 [band0, band1, ...]，每项 (N, C, T) float32。

    分块处理以控制内存峰值（sosfiltfilt 内部会分配工作缓冲）。
    """
    out = []
    for f1, f2 in zip(band_edges[:-1], band_edges[1:]):
        parts = []
        for s in range(0, len(X), chunk):
            parts.append(bandpass(X[s:s + chunk], f1, f2, sfreq, order))
        out.append(np.concatenate(parts, axis=0).astype(np.float32))
    return out
