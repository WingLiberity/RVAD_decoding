# -*- coding: utf-8 -*-
"""模型工厂与注册表。"""
from .eegnet import EEGNet
from .deepconvnet import DeepConvNet
from .fbchnet import FBCNet
from .svm import build_svm

# 支持的模型（训练脚本 --models 选项取值）
MODEL_REGISTRY = ('svm', 'eegnet', 'deepconvnet', 'fbchnet')


def create_model(name, n_channels, n_times, n_classes, cfg):
    """根据名称创建深度学习模型（SVM 不走此函数）。

    Args:
        name: 'eegnet' / 'deepconvnet' / 'fbchnet'
        cfg:  配置模块（含各模型超参与 SFREQ）
    """
    if name == 'eegnet':
        p = cfg.EEGNET
        return EEGNet(n_channels, n_times, n_classes,
                      F1=p['F1'], D=p['D'],
                      kernel_1=p['kernel_1'], kernel_2=p['kernel_2'],
                      sfreq=cfg.SFREQ, drop_rate=p['drop_rate'])
    if name == 'deepconvnet':
        p = cfg.DEEPCONVNET
        return DeepConvNet(n_channels, n_times, n_classes,
                           F=p['F'], conv_temporal=p['conv_temporal'],
                           drop_rate=p['drop_rate'])
    if name == 'fbchnet':
        p = cfg.FBCNET
        return FBCNet(n_channels, n_times, n_bands=len(p['band_edges']) - 1,
                      n_classes=n_classes,
                      spatial_filters=p['spatial_filters'],
                      num_windows=p['num_windows'], drop_rate=p['drop_rate'])
    raise ValueError(f'未知模型: {name}')


__all__ = ['MODEL_REGISTRY', 'create_model', 'build_svm']
