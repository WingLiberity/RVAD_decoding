# -*- coding: utf-8 -*-
"""FBCNet（Mane et al., 2021）PyTorch 实现。

滤波组在数据预处理阶段完成（见 preprocessing.filter_bands），
模型输入为 (N, n_bands, C, T)，前向依次经过：
  逐频带空间卷积 -> 方差池化 -> 拼接 -> 分类器。
"""
import torch
import torch.nn as nn

from utils import init_weights


class FBCNet(nn.Module):
    def __init__(self, n_channels, n_times, n_bands, n_classes=2,
                 spatial_filters=32, num_windows=8, drop_rate=0.3):
        super().__init__()
        self.n_bands = n_bands
        self.num_windows = num_windows
        self.n_channels = n_channels

        self.spatial_conv = nn.Conv2d(1, spatial_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm1d(spatial_filters * n_bands)  # 作用于拼接后的 (N, S*B, W)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_filters * n_bands * num_windows, n_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        # x: (N, B, C, T)
        n_batch = x.size(0)
        feats = []
        for b in range(self.n_bands):
            h = x[:, b].unsqueeze(1)          # (N, 1, C, T)
            h = self.spatial_conv(h).squeeze(2)  # (N, S, T)
            t = h.size(-1)
            win = t // self.num_windows       # 每窗样本数（丢弃尾部不足部分）
            h = h[..., :win * self.num_windows].view(n_batch, -1, self.num_windows, win)
            h = torch.var(h, dim=-1)          # 方差池化 (N, S, W)
            feats.append(h)
        x = torch.cat(feats, dim=1)           # (N, S*B, W)
        x = self.bn(x)
        x = self.drop(x)
        return self.classifier(x)
