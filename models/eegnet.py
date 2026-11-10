# -*- coding: utf-8 -*-
"""EEGNet（Lawhern et al., 2018）PyTorch 实现。

输入: (N, 1, C, T)；核长按采样率从 128Hz 参考值缩放。
"""
import torch
import torch.nn as nn

from utils import init_weights


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, F1=8, D=2,
                 kernel_1=64, kernel_2=16, sfreq=128, drop_rate=0.25):
        super().__init__()
        # 时间卷积核长按采样率缩放，保持时间跨度与 128Hz 参考一致
        k1 = max(8, int(round(kernel_1 * sfreq / 128)))
        k2 = max(4, int(round(kernel_2 * sfreq / 128)))
        F2 = F1 * D

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),  # 深度卷积(空间)
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_rate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, k2), padding=(0, k2 // 2), bias=False),  # 深度卷积(时间)
            nn.Conv2d(F2, F2, (1, 1), bias=False),                          # 逐点卷积
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_rate),
        )

        # 通过一次前向计算分类器输入维度（适配任意输入长度）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block2(self.block1(dummy))
            n_features = int(out.numel())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, n_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
