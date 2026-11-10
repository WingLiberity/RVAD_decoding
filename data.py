# -*- coding: utf-8 -*-
"""数据加载：遍历数据集下所有 pp_eeg/*.npz，拼接为 (N, C, T) 与标签。"""
from pathlib import Path
import numpy as np


def _subject_npz(subject_dir, subdir='pp_eeg', task_filter=None):
    """返回某被试目录下、满足任务筛选的 npz 文件列表。

    S7 在同一天做了两次按键任务，只发布第一次（与 BIDS 发布一致），
    因此跳过 ``*_20250625_2.npz``。
    """
    files = sorted(f for f in Path(subject_dir).rglob('*.npz')
                   if f.parent.name == subdir
                   and not f.stem.endswith('_20250625_2')
                   and (task_filter is None or task_filter in f.parts))
    return files


def list_subjects(data_dir, subdir='pp_eeg', task_filter=None):
    """枚举有数据的被试，返回 [(组名, 被试名, 被试目录), ...]。

    task_filter: 只保留任务路径中含该关键字的文件（如 'TC_AJ' 塔材按键）。
    """
    subjects = []
    for grp in sorted(Path(data_dir).iterdir()):
        if not grp.is_dir():
            continue
        for subj in sorted(grp.iterdir()):
            if not subj.is_dir():
                continue
            if _subject_npz(subj, subdir, task_filter):
                subjects.append((grp.name, subj.name, subj))
    return subjects


def load_subject(subject_dir, subdir='pp_eeg', task_filter=None,
                 max_samples=None, avg_channels=False):
    """加载单个被试（可能包含多个任务/多个文件）的数据并拼接。

    Args:
        subject_dir: 被试目录，如 .../expert/S1。
        subdir: 预处理数据子目录名。
        task_filter: 只加载任务路径中含该关键字的文件，None 表示全部。
        max_samples: 仅取前 max_samples 条样本（调试用），None 表示全部。
        avg_channels: True 时每条 trial 将全部通道取平均为单通道 ERP 波形，
                      X 形状为 (N, 1, n_times)；False 保留全部通道 (N, C, T)。

    Returns:
        X: (N, n_channels, n_times) float32
        y: (N,) int64
        files: 参与拼接的文件列表
    """
    files = _subject_npz(subject_dir, subdir, task_filter)
    if not files:
        raise FileNotFoundError(f'{subject_dir} 下没有匹配 task_filter={task_filter} '
                                f'的 {subdir}/*.npz 文件')

    xs, ys = [], []
    for f in files:
        d = np.load(f)
        xs.append(d['epochs'].astype(np.float32))
        ys.append(d['labels'].astype(np.int64))
        if max_samples is not None and sum(len(x) for x in xs) >= max_samples:
            break

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]
    if avg_channels:
        # 每条 trial 取全部通道均值，得到单通道 ERP 波形
        X = X.mean(axis=1, keepdims=True).astype(np.float32)
    return X, y, files


def load_all(data_dir, subdir='pp_eeg', max_samples=None, avg_channels=False):
    """加载全部 .npz 文件并拼接。

    Args:
        data_dir: 数据集根目录。
        subdir:   预处理数据子目录名，默认 'pp_eeg'。
        max_samples: 仅取前 max_samples 条样本（调试用），None 表示全部。
        avg_channels: True 时每条 trial 将全部通道取平均为单通道 ERP 波形，
                      X 形状为 (N, 1, n_times)；False 保留全部通道 (N, C, T)。

    Returns:
        X: (N, n_channels, n_times) float32
        y: (N,) int64
        files: 参与拼接的文件列表
    """
    files = sorted(f for f in Path(data_dir).rglob('*.npz') if f.parent.name == subdir)
    if not files:
        raise FileNotFoundError(f'未在 {data_dir} 下找到任何 {subdir}/*.npz 文件')

    xs, ys = [], []
    for f in files:
        d = np.load(f)
        xs.append(d['epochs'].astype(np.float32))
        ys.append(d['labels'].astype(np.int64))
        if max_samples is not None and sum(len(x) for x in xs) >= max_samples:
            break

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]
    if avg_channels:
        # 每条 trial 取全部通道均值，得到单通道 ERP 波形
        X = X.mean(axis=1, keepdims=True).astype(np.float32)
    return X, y, files
