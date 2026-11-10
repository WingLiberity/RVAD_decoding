# -*- coding: utf-8 -*-
"""SVM：直接对（通道平均后的）ERP 波形分类。"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(C=1.0, kernel='rbf'):
    """构建 标准化 -> RBF-SVM 管道。

    输入为 (N, n_times) 的 ERP 特征（由训练脚本展平），输出类别预测。
    """
    return make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel=kernel, gamma='scale', class_weight='balanced'),
    )
