# -*- coding: utf-8 -*-
"""deepConvNet（Schirrmeister et al., 2017）PyTorch 实现。

输入: (N, 1, C, T)；5 个卷积块，滤波器数 25/25/50/100/200/200。
"""
import torch
import torch.nn as nn

from utils import init_weights


class DeepConvNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, F=25,
                 conv_temporal=10, drop_rate=0.5):
        super().__init__()
        pad = conv_temporal // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, F, (1, conv_temporal), padding=(0, pad), bias=False),
            nn.Conv2d(F, F, (n_channels, 1), bias=False),   # 空间卷积
            nn.BatchNorm2d(F),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(drop_rate),
            *self._block(F, 2 * F, conv_temporal, drop_rate),
            *self._block(2 * F, 4 * F, conv_temporal, drop_rate),
            *self._block(4 * F, 8 * F, conv_temporal, drop_rate),
            *self._block(8 * F, 8 * F, conv_temporal, drop_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.features(dummy)
            n_features = int(out.numel())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, n_classes),
        )
        self.apply(init_weights)

    @staticmethod
    def _block(in_ch, out_ch, k, drop_rate):
        pad = k // 2
        return [
            nn.Conv2d(in_ch, out_ch, (1, k), padding=(0, pad), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(drop_rate),
        ]

    def forward(self, x):
        return self.classifier(self.features(x))
