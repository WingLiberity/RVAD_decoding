# Single-trial decoding of target vs. non-target RSVP trials

Reference implementation for the decoding analyses of the **RVAD** dataset
(RSVP-based visual attention dataset, BIDS 1.9.0).

Four classifiers are trained **separately for each participant** on single-trial
EEG epochs and evaluated on a held-out test split:

| key | model | reference |
|---|---|---|
| `svm` | linear/RBF support vector machine on the flattened ERP | — |
| `eegnet` | EEGNet | Lawhern et al., *J. Neural Eng.* 15:056013 (2018) |
| `deepconvnet` | DeepConvNet | Schirrmeister et al., *Hum. Brain Mapp.* 38:5391 (2017) |
| `fbchnet` | FBCNet | Mane et al., *IEEE Trans. Neural Syst. Rehabil. Eng.* 29:2344 (2021) |

## Repository layout

| path | purpose |
|---|---|
| `train_cv.py` | entry point: per-participant split, training, evaluation, result tables |
| `config.py` | all hyper-parameters (data paths, split, optimiser, per-model settings) |
| `data.py` | reads the `.npz` epochs written by `bids_to_npz.py` |
| `preprocessing.py` | z-scoring and the filter bank used by FBCNet |
| `metrics.py` | accuracy, balanced accuracy, F1 and ROC-AUC |
| `utils.py` | seeding, device selection, class weights, weight initialisation |
| `models/` | one module per classifier; `__init__.py` is the model registry |
| `bids_to_npz.py` | converts the BIDS derivative (`pp-eeg-v2`) to `.npz` |
| `summarize.py` | merges the per-model CSVs into per-participant tables |
| `results/reference/` | outputs reproduced on the released dataset |

Every analysis is participant-specific: one model is fitted per recording, never
pooled across participants. `train_cv.py --models` accepts any subset of
`svm eegnet deepconvnet fbchnet`.

## Data

The code runs on the published dataset and its preprocessed derivative.

* Raw BIDS dataset: `RVAD_BIDS/` (188 files, 15 participants, 23 recordings).
* Preprocessed single-trial epochs: `RVAD_BIDS/derivatives/pp-eeg-v2/`
  (EEGLAB `.set`/`.fdt`, 64 channels, 1,060 epochs per recording,
  −200…800 ms at 1000 Hz, 1–40 Hz zero-phase FIR, units µV).

> **Dataset citation.** <AUTHORS>. *RVAD: an RSVP-based visual attention dataset
> with expert, junior and novice performers* (2026). Dataset DOI: `<DATASET DOI>`.

Decoding uses the **keypress** task (`TC_AJ`) only, i.e. 15 recordings, one per
participant (the second session of S7 is not part of the release).

### Preparing the data

`train_cv.py` reads compressed `.npz` files; convert the BIDS derivative once:

```bash
python bids_to_npz.py \
    --bids /path/to/RVAD_BIDS/derivatives/pp-eeg-v2 \
    --out  ./data
```

Resulting layout (group names follow `participants.tsv`):

```
data/
└── <group>/                 expert | junior | novice
    └── sub-S1/
        └── TC_AJ/           TC_AJ = keypress, TC_MS = counting
            └── pp_eeg_v2/
                └── sub-S1_keypress.npz     keys: epochs (1060, 64, 1001), labels (1060,)
```

`labels` follows the BIDS `trial_type`: `0 = non-target (standard)`,
`1 = target`. Channel order is identical to the BIDS derivative; **Oz is channel
index 56** (0-based, O1 = 57, O2 = 58, ECG = 59, HEOR/HEOL/VEOU/VEOL = 60–63).
Note that trials are stored targets first, so always select by label, never by
row index.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is used for the deep models; install the build matching your CUDA
version from <https://pytorch.org> if you need GPU support.

## Usage

```bash
# all four models, all participants
python train_cv.py

# a single model
python train_cv.py --models svm --verbose

# point to another data folder, change the seed
RVAD_DATA_DIR=/data/rvad python train_cv.py --seed 42

# merge per-model results into per-participant tables
python summarize.py --results results
```

## Pipeline

1. **Input.** Each epoch (−200…800 ms, 1001 samples at 1000 Hz) is averaged
   across the 64 electrodes (`config.AVG_CHANNELS = True`), giving a
   one-dimensional waveform per trial. Set `AVG_CHANNELS = False` to keep all
   channels.
2. **Split.** Per participant, stratified 80/10/10 (train/validation/test),
   `random_state = 42`. The validation split is only used for early stopping of
   the deep models.
3. **Normalisation.** z-scoring with statistics from the training split only
   (`preprocessing.zscore_fit/apply`), no information leakage.
4. **SVM.** `StandardScaler → SVC(kernel='rbf', C=1, gamma='scale',
   class_weight='balanced')` on the flattened waveform; AUC from
   `decision_function`.
5. **Deep models.** Adam (lr 1e-3, weight decay 1e-4), batch size 64, class
   weighted cross-entropy, at most 30 epochs, early stopping on validation
   balanced accuracy (patience 8). FBCNet uses four bands
   (4–8–12–16–20 Hz).
6. **Metrics.** Accuracy, balanced accuracy, F1 (target = positive class) and
   ROC-AUC on the test split, summarized across participants as mean ± SD.

## Results

`results/reference/` contains the outputs reproduced on the released dataset
(15 participants, keypress task, 80/10/10 split, seed 42):

| model | accuracy | balanced accuracy | F1 | AUC |
|---|---|---|---|---|
| SVM | 0.824 ± 0.103 | 0.720 ± 0.136 | 0.519 ± 0.218 | 0.798 ± 0.165 |
| EEGNet | 0.782 ± 0.113 | 0.738 ± 0.121 | 0.510 ± 0.176 | 0.816 ± 0.145 |
| DeepConvNet | 0.776 ± 0.143 | 0.722 ± 0.104 | 0.492 ± 0.151 | 0.789 ± 0.128 |
| FBCNet | 0.720 ± 0.169 | 0.664 ± 0.144 | 0.436 ± 0.201 | 0.719 ± 0.169 |

Per-participant values: `results/reference/per_participant_decoding.csv`
(wide) and `per_participant_decoding_long.csv` (long). Because the two classes
are imbalanced (900 non-target vs. 160 target trials per recording), balanced
accuracy and AUC are the metrics to compare; accuracy alone is inflated.

## Outputs

```
results/
├── svm_subject.csv                one row per participant and model
├── eegnet_subject.csv
├── deepconvnet_subject.csv
├── fbchnet_subject.csv
├── per_participant_decoding.csv   wide table (all models side by side)
├── per_participant_decoding_long.csv
├── decode_metrics.csv
└── reference/                     outputs shipped with the release
```

## Configuration

All hyper-parameters live in `config.py`; `DATA_DIR` and `RESULTS_DIR` can also
be set with the environment variables `RVAD_DATA_DIR` and `RVAD_RESULTS_DIR`.

## Reproducibility notes

* Single 80/10/10 split, seed 42; no cross-validation. With ~106 test trials per
  participant (≈16 of them targets) the per-participant estimates are noisy —
  S5, S7 and S12 are close to chance for all four models.
* Deep models are seeded (`utils.set_seed`), but GPU non-determinism can cause
  small run-to-run differences.
* Artifact handling is not part of this repository: the released epochs are
  unfiltered for artifacts, and the per-recording quality metrics (bad channels,
  epochs above 100 µV, baseline RMS) are provided as `qc_metrics.csv` /
  `sme_metrics.csv` in the dataset root.

## Citation

If you use this code, please cite the dataset and the methods it builds on:

```bibtex
@dataset{rvad2026,
  title  = {RVAD: an RSVP-based visual attention dataset with expert, junior and novice performers},
  author = {<AUTHORS>},
  year   = {2026},
  doi    = {<DATASET DOI>}
}
```

See `CITATION.cff` for machine-readable metadata.

## License

MIT — see `LICENSE`.
