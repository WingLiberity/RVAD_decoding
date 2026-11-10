# -*- coding: utf-8 -*-
"""分类评估指标。"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)


def evaluate(y_true, y_pred, y_score=None):
    """计算常用二分类指标。

    Args:
        y_true:  真实标签 (N,)
        y_pred:  预测标签 (N,)
        y_score: 预测分数/正类概率 (N,)，用于 AUC；可为 None

    Returns:
        dict: acc / bal_acc / f1 / auc
    """
    res = {
        'acc': accuracy_score(y_true, y_pred),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            res['auc'] = roc_auc_score(y_true, y_score)
        except ValueError:  # 该折测试集只有单一类别时
            res['auc'] = float('nan')
    return res


def format_metric(values):
    """将一列 fold 指标格式化为 mean±std 字符串（忽略 NaN）。"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 'NaN'
    return f'{v.mean():.4f}±{v.std():.4f}'
