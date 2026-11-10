# -*- coding: utf-8 -*-
"""通用工具：随机种子、设备、权重初始化。"""
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """固定所有随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu=True):
    """返回计算设备。"""
    if gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def init_weights(m):
    """Xavier 初始化卷积/全连接层。"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def class_weights(y, n_classes=2):
    """由类别频率计算加权损失权重（类别数/（类别数×频率））。"""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(n_classes / (counts * n_classes), dtype=torch.float32)
