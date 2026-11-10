# -*- coding: utf-8 -*-
"""Convert the public BIDS derivative (pp_eeg_v2, EEGLAB .set/.fdt) to .npz files.

The training scripts expect this layout::

    data/<group>/<subject>/<task>/pp_eeg_v2/<record>.npz   keys: epochs (N,C,T), labels (N,)

Example::

    python bids_to_npz.py --bids /path/to/RVAD_BIDS/derivatives/pp-eeg-v2 --out ./data

The .set files are read with scipy (header) + the accompanying .fdt binary, so
MNE is not required and the non-unique event latencies of the exported epochs
are not an issue.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio

TASK = {'keypress': 'TC_AJ', 'counting': 'TC_MS'}


def read_set(set_path):
    """Return (data (trials, channels, samples), meta) from an EEGLAB .set/.fdt pair."""
    eeg = sio.loadmat(set_path, simplify_cells=True)['EEG']
    n_ch, n_pt, n_tr = int(eeg['nbchan']), int(eeg['pnts']), int(eeg['trials'])
    fdt = Path(set_path).with_suffix('.fdt')
    x = np.fromfile(fdt, dtype='<f4').reshape(n_tr, n_pt, n_ch).transpose(0, 2, 1)
    ch_names = [c['labels'] for c in eeg['chanlocs']]
    return np.ascontiguousarray(x, dtype=np.float32), ch_names, float(eeg['srate'])


def read_labels(events_tsv):
    """1 = target, 0 = non-target, taken from the trial_type column."""
    with open(events_tsv, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    return np.array([1 if r['trial_type'] == 'target' else 0 for r in rows], dtype=np.int64)


def read_groups(bids):
    """participant -> group, from participants.tsv."""
    f = Path(bids).parent.parent / 'participants.tsv'
    if not f.exists():
        f = Path(bids) / 'participants.tsv'
    if not f.exists():
        return {}
    with open(f, newline='', encoding='utf-8-sig') as fh:
        return {r['participant_id']: r.get('group', 'unknown') for r in csv.DictReader(fh, delimiter='\t')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids', required=True, help='derivatives/pp-eeg-v2 folder')
    ap.add_argument('--out', default='data', help='output folder for the .npz tree')
    args = ap.parse_args()

    bids, out = Path(args.bids), Path(args.out)
    groups = read_groups(bids)
    sets = sorted(bids.glob('sub-*/eeg/*_eeg.set'))
    if not sets:
        raise SystemExit(f'no *_eeg.set found under {bids}')

    for sp in sets:
        sub = sp.name.split('_')[0]                       # sub-S1
        task = sp.name.split('_')[2].replace('task-', '')  # keypress / counting
        x, ch_names, sfreq = read_set(sp)
        y = read_labels(sp.with_name(sp.name.replace('_eeg.set', '_events.tsv')))
        if len(y) != x.shape[0]:
            raise SystemExit(f'{sp.name}: {len(y)} events vs {x.shape[0]} epochs')
        if ch_names[56] != 'Oz':
            raise SystemExit(f'{sp.name}: channel 56 is {ch_names[56]}, expected Oz')

        dst = out / groups.get(sub, 'unknown') / sub / TASK.get(task, task) / 'pp_eeg_v2'
        dst.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst / f'{sub}_{task}.npz', epochs=x, labels=y)
        print(f'{sp.name:52s} -> {dst}  {x.shape}  target={int(y.sum())}  sfreq={sfreq:g}')


if __name__ == '__main__':
    main()
