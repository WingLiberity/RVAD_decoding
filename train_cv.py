# -*- coding: utf-8 -*-
"""逐人训练主脚本（80% 训练 / 10% 验证 / 10% 测试）。

每个被试单独加载数据（默认只用"塔材按键"TC_AJ 任务），
按 80/10/10 分层划分后独立训练与测试，再对全部被试汇总 mean±std。

用法示例:
    python train_cv.py                          # 训练全部 4 个模型
    python train_cv.py --models eegnet fbchnet  # 只训练指定模型
    python train_cv.py --epochs 30              # 指定最大轮数
    python train_cv.py --no-gpu                 # 强制使用 CPU
    python train_cv.py --max-samples 2000       # 调试：每人只取前 2000 条样本

结果保存到 results/ 下（每模型 per-subject CSV + 汇总 CSV）。
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from data import list_subjects, load_subject
from metrics import evaluate, format_metric
from models import MODEL_REGISTRY, create_model, build_svm
from preprocessing import filter_bands, zscore_apply, zscore_fit
from utils import class_weights, get_device, set_seed

METRIC_COLS = ['acc', 'bal_acc', 'f1', 'auc']


class EEGDataset(Dataset):
    """EEG 数据集。

    X: (N, C, T)；bands: [band0(N,C,T), ...]（FBCNet 使用，此时 X 可为 None）。
    __getitem__ 返回 (1,C,T) 或 (B,C,T) 张量 + 标签。
    """

    def __init__(self, X, y, bands=None):
        self.X = X
        self.bands = bands
        self.y = np.asarray(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.bands is not None:
            x = np.stack([b[i] for b in self.bands]).astype(np.float32)  # (B,C,T)
        else:
            x = self.X[i:i + 1]                                          # (1,C,T)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    """返回 (y_true, y_pred, y_score)，y_score 为正类概率。"""
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ys.append(yb.numpy())
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(probs)
    return y_true, (y_score >= 0.5).astype(int), y_score


def split_subject(X, y, seed):
    """将单个被试数据按 80/10/10 分层划分为 训练/验证/测试 索引。"""
    n = len(y)
    rest_idx, test_idx = train_test_split(
        np.arange(n), test_size=cfg.SPLIT[2], stratify=y, random_state=seed)
    # 剩余 90% 中再切出验证集（占总体 10% = 90% 的 1/9）
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=cfg.SPLIT[1] / (cfg.SPLIT[0] + cfg.SPLIT[1]),
        stratify=y[rest_idx], random_state=seed + 1)
    return train_idx, val_idx, test_idx


def train_torch_subject(name, X, y, train_idx, val_idx, test_idx, args, device):
    """训练一个深度学习模型（单被试），返回测试集指标。"""
    n_classes = int(y.max()) + 1

    # 逐通道 z-score（统计量仅来自训练集，避免信息泄漏）
    mean, std = zscore_fit(X[train_idx])
    Xtr = zscore_apply(X[train_idx], mean, std)
    Xval = zscore_apply(X[val_idx], mean, std)
    Xte = zscore_apply(X[test_idx], mean, std)

    bands = None
    if name == 'fbchnet':
        band_edges = cfg.FBCNET['band_edges']
        bands = dict(
            tr=filter_bands(Xtr, band_edges, cfg.SFREQ),
            val=filter_bands(Xval, band_edges, cfg.SFREQ),
            te=filter_bands(Xte, band_edges, cfg.SFREQ),
        )
        del Xtr, Xval, Xte  # 释放原始信号，控制内存
        Xtr = Xval = Xte = None

    model = create_model(name, X.shape[1], X.shape[2], n_classes, cfg)
    model.to(device)

    tr_ds = EEGDataset(Xtr, y[train_idx], bands['tr'] if bands else None)
    val_ds = EEGDataset(Xval, y[val_idx], bands['val'] if bands else None)
    te_ds = EEGDataset(Xte, y[test_idx], bands['te'] if bands else None)

    loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                                 weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y[train_idx], n_classes).to(device))

    # 训练 + 早停（依据验证集 balanced accuracy）
    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, loader, criterion, optimizer, device)
        yt, yp, ys = predict(model, val_loader, device)
        val_m = evaluate(yt, yp, ys)
        if val_m['bal_acc'] > best_val:
            best_val = val_m['bal_acc']
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                break
        if args.verbose:
            print(f'      epoch {epoch:2d}/{args.epochs} | '
                  f'tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} '
                  f'val_bal_acc={val_m["bal_acc"]:.4f}')

    model.load_state_dict(best_state)
    yt, yp, ys = predict(model, te_loader, device)
    m = evaluate(yt, yp, ys)
    m['val_bal_acc'] = best_val
    return m


def svm_subject(X, y, train_idx, test_idx, args):
    """SVM（单被试）：将 ERP 波形展平为特征后直接分类。"""
    mean, std = zscore_fit(X[train_idx])
    Xtr = zscore_apply(X[train_idx], mean, std).reshape(len(train_idx), -1)
    Xte = zscore_apply(X[test_idx], mean, std).reshape(len(test_idx), -1)

    p = cfg.SVM
    pipe = build_svm(C=p['svm_C'], kernel=p['kernel'])
    pipe.fit(Xtr, y[train_idx])
    y_score = pipe.decision_function(Xte)
    y_pred = pipe.predict(Xte)
    return evaluate(y[test_idx], y_pred, y_score)


def run_subject(name, group, subject, subject_dir, args, device):
    """对单个被试运行指定模型：加载 -> 80/10/10 划分 -> 训练/测试。"""
    X, y, files = load_subject(subject_dir, cfg.DATA_SUBDIR, cfg.TASK_FILTER,
                               args.max_samples, avg_channels=cfg.AVG_CHANNELS)
    train_idx, val_idx, test_idx = split_subject(X, y, args.seed)

    if name == 'svm':
        m = svm_subject(X, y, train_idx, test_idx, args)
    else:
        m = train_torch_subject(name, X, y, train_idx, val_idx, test_idx,
                                args, device)

    m = {'model': name, 'group': group, 'subject': subject,
         'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
         **m}
    return m


def save_results(name, rows, outdir):
    """保存单模型 per-subject 指标。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = ['model', 'group', 'subject', 'n_train', 'n_val', 'n_test',
            *METRIC_COLS, 'val_bal_acc']
    with open(outdir / f'{name}_subject.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return cols


def main():
    ap = argparse.ArgumentParser(description='RVAD EEG 逐人 80/10/10 训练')
    ap.add_argument('--models', nargs='+', choices=MODEL_REGISTRY,
                    default=list(MODEL_REGISTRY), help='要训练的模型')
    ap.add_argument('--epochs', type=int, default=cfg.EPOCHS, help='每个被试最大训练轮数')
    ap.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE)
    ap.add_argument('--seed', type=int, default=cfg.RANDOM_STATE)
    ap.add_argument('--no-gpu', action='store_true', help='强制使用 CPU')
    ap.add_argument('--outdir', type=str, default=str(cfg.RESULTS_DIR),
                    help='结果输出目录')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='调试用：每人仅取前 N 条样本')
    ap.add_argument('--verbose', action='store_true', help='打印每个 epoch 的进度')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu') if args.no_gpu else get_device()
    print(f'设备: {device}')

    subjects = list_subjects(cfg.DATA_DIR, cfg.DATA_SUBDIR, cfg.TASK_FILTER)
    if not subjects:
        raise SystemExit(f'未找到满足 task_filter={cfg.TASK_FILTER} 的被试')
    print(f'被试数: {len(subjects)} | 任务筛选: {cfg.TASK_FILTER} | '
          f'划分: {cfg.SPLIT} (训练/验证/测试)')

    summary = {}
    for name in args.models:
        print(f'===== {name} =====')
        rows = []
        for gi, (group, subject, subject_dir) in enumerate(subjects, 1):
            m = run_subject(name, group, subject, subject_dir, args, device)
            rows.append(m)
            print(f'  [{name}] {group}/{subject} ({gi}/{len(subjects)}) | '
                  + ' '.join(f'{k}={v:.4f}' for k, v in m.items()
                             if k in METRIC_COLS)
                  + f' | n_train={m["n_train"]}')
        cols = save_results(name, rows, args.outdir)
        metric_cols = [c for c in cols if c in METRIC_COLS]
        summary[name] = {c: format_metric([r[c] for r in rows])
                         for c in metric_cols}
        for k, v in summary[name].items():
            print(f'  {k}: {v}')

    # 汇总（控制台 + CSV）
    print('\n===== 汇总 (mean±std over subjects) =====')
    header = ['model', *METRIC_COLS]
    print('  ' + ' | '.join(header))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'summary.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for name in args.models:
            row = {'model': name, **summary[name]}
            w.writerow(row)
            print('  ' + ' | '.join(str(row[c]) for c in header))
    print(f'\n结果已保存至: {outdir}')


if __name__ == '__main__':
    main()
