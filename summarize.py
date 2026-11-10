# -*- coding: utf-8 -*-
"""Merge the per-model, per-participant CSVs into one long and one wide table.

Run after ``train_cv.py``:

    python summarize.py --results results
"""
import argparse
from pathlib import Path

import pandas as pd

MODELS = ['svm', 'eegnet', 'deepconvnet', 'fbchnet']
GROUP = {'expert': 'expert', 'novice': 'junior', 'non-performance': 'novice'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results', help='folder with <model>_subject.csv')
    args = ap.parse_args()
    res = Path(args.results)

    frames = []
    for m in MODELS:
        f = res / f'{m}_subject.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d['group'] = d['group'].map(lambda g: GROUP.get(g, g))
        d['model'] = m
        frames.append(d)
    if not frames:
        raise SystemExit(f'no <model>_subject.csv found in {res}')

    df = pd.concat(frames, ignore_index=True)
    long = df[['model', 'group', 'subject', 'n_train', 'n_val', 'n_test',
               'acc', 'bal_acc', 'f1', 'auc']]
    long.to_csv(res / 'per_participant_decoding_long.csv', index=False, encoding='utf-8')

    metrics = _agg(long, ['group', 'model'], ['acc', 'bal_acc', 'f1', 'auc']).reset_index()
    metrics.insert(2, 'n_subjects', long.groupby(['group', 'model']).size().values)
    metrics.to_csv(res / 'decode_metrics.csv', index=False, encoding='utf-8')

    wide = long.pivot_table(index=['group', 'subject'], columns='model',
                            values=['acc', 'bal_acc', 'auc'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide.reset_index()
    order = ['expert', 'junior', 'novice']
    wide['group'] = pd.Categorical(wide['group'], order, ordered=True)
    wide = wide.sort_values(['group', 'subject']).reset_index(drop=True)
    wide.to_csv(res / 'per_participant_decoding.csv', index=False, encoding='utf-8')

    print(wide.round(4).to_string(index=False))
    print()
    cols = ['acc', 'bal_acc', 'f1', 'auc']
    print(_agg(long, ['group', 'model'], cols).to_string())
    print()
    print(_agg(long, ['model'], cols).to_string())


def _agg(df, keys, cols):
    """mean and population SD (ddof=0), as reported in the paper."""
    m = df.groupby(keys)[cols].mean()
    m.columns = [f'{c}_mean' for c in m.columns]
    s = df.groupby(keys)[cols].agg(lambda x: x.std(ddof=0))
    s.columns = [f'{c}_std' for c in s.columns]
    return pd.concat([m, s], axis=1).round(4)


if __name__ == '__main__':
    main()
