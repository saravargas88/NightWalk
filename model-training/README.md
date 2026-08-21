# model-training

This directory contains all training, evaluation, and baseline scripts for **NightWalk** — predicting nighttime street luminance from daytime Street View Images (SVI).

---

## How to train the model (start here)

`run_experiments.py` is the main entry point — it fine-tunes the brightness regression model across all backbone conditions (ImageNet, DINO-counts, SSL) and writes a unified results summary. It has two prerequisites that must exist **before** you run it:

**1. Train/test splits** (`splits/train_split.csv`, `splits/test_split.csv`)
```bash
cd splits
python make_split.py
```
This creates a geographically spread test split from the day–night pairs. Run once — `run_experiments.py` checks for these files and exits immediately with an error telling you to run this if they're missing.

**2. DINO feature counts** (needed for the `dino_counts` backbone condition)
```bash
cd dino_feature_detection
python dino_exps.py
```
Detects streetlights/trees/storefronts in daytime images. Then pretrain the backbone on those counts:
```bash
python model-training/pretraining/train_efficientnet_multihead.py
```
This produces `best_efficientnet_multihead.pt`, the "dino_counts" backbone checkpoint. (If you only care about the `imagenet` or `ssl` backbone conditions, you can skip this step.)

**3. Run the experiments**
```bash
# Full sweep across all three backbones
python model-training/run_experiments.py

# Just imagenet + dino_counts, skipping SSL pretraining if you don't need it
python model-training/run_experiments.py --backbones imagenet dino_counts

# SSL backbone will self-pretrain automatically on first run (or pass --skip-ssl-pretrain
# if you already have a checkpoint at ssl-pretrain/best_ssl_backbone.pt)
```
Results land in `model-training/finetune-runs/<backbone>/n<N>/` per fold, with a summary at `model-training/results_summary.csv`.

**In short:** `splits/make_split.py` → `dino_feature_detection/dino_exps.py` → `pretraining/train_efficientnet_multihead.py` → `run_experiments.py`.

For the classification ablation (predicting a brightness bin instead of a continuous score), the same idea applies but with `run_experiments_new.py` instead of `run_experiments.py` — see "Running other experiments" below.

---

## Directory layout

```
model-training/
├── run_experiments.py          ← regression ablation orchestrator (backbone × n_train) — main entry point
├── run_experiments_new.py      ← classification ablation orchestrator (backbone × n_train)
│
├── pretraining/                ← train the backbone before brightness fine-tuning
│   ├── train_efficientnet_multihead.py   Trains EfficientNet on DINO counts
│   │                                     (tree / streetlight / storefront). Produces
│   │                                     best_efficientnet_multihead.pt ("dino_counts" backbone)
│   ├── pretrain_selfsupervised.py        SimCLR self-supervised pretraining on 13k day images.
│   │                                     Produces ssl-pretrain/best_ssl_backbone.pt
│   └── train_counts_small.py             Older / lighter version of multihead training.
│                                         Kept as reference; superseded by multihead.
│
├── regression/                 ← predict a continuous brightness score
│   ├── train_brightness_score.py  Multi-target regressor (gray_mean, luma_mean, value_mean,
│   │                               gray_mean_zscore simultaneously). Simple 80/20 split.
│   │                               Outputs → brightness-regression-run/
│   └── finetune_brightness.py     Single-target regressor with 5-fold cross-validation.
│                                   Called by run_experiments.py. Supports imagenet /
│                                   dino_counts / ssl backbone. Outputs → finetune-runs/
│
├── classification/             ← predict 1-of-4 brightness bins
│   └── train_brightness_class.py  4-class classifier (very_dark / dark / bright / very_bright).
│                                   Bins are quartiles of gray_mean computed from the train
│                                   split only (no leakage). Called by run_experiments_new.py.
│                                   Outputs → brightness-class-runs/
│
└── eval/                       ← evaluation and analysis
    ├── eval_brightness_checkpoint.py   Evaluate train_brightness_score.py checkpoint on test split.
    ├── linear_probe.py                 Frozen EfficientNet embedding + Ridge regression baseline.
    │                                   Supports --extra-features to add DINO counts + bbox areas.
    │                                   Directly comparable to finetune_brightness.py results.
    ├── visualize_training.py           Plots per-target MAE curves from multihead training logs.
    ├── predict_night_brightness.py     Early-stage inference script (kept for reference).
    └── train_brightness_levels_archive.py  Older classification script superseded by
                                            classification/train_brightness_class.py.
```

---

## Data flow

```
urban-mosaic/washington-square/     ~100k daytime SVI images (13k used for backbone)
        ↓
pretraining/train_efficientnet_multihead.py
        ↓
best_efficientnet_multihead.pt      "dino_counts" backbone checkpoint
        ↓
regression/finetune_brightness.py   (called by run_experiments.py, 5-fold CV)
        ↓
finetune-runs/<backbone>/n<N>/fold_<k>/best_model.pt
        ↓
results_summary.csv                 → aggregated val + test metrics
```

Day-night pairs live in:
```
splits/train_split.csv    ~780 pairs
splits/test_split.csv     ~200 pairs
labeling/brightness_metrics/experiment_outputs/paired_dataset_with_brightness.csv
    → gray_mean, luma_mean, value_mean, gray_mean_zscore + bbox/DINO features per pair
```

---

## Running other experiments

### Best single regression model (multi-target, simple 80/20 split)
```bash
python model-training/regression/train_brightness_score.py \
    --epochs 45 --lr-backbone 1e-5 --lr-head 1e-4
```
Evaluate on test split:
```bash
python model-training/eval/eval_brightness_checkpoint.py \
    --image-dir urban-mosaic/washington-square
```

### Linear probe baseline
```bash
# Embedding only
python model-training/eval/linear_probe.py --backbone imagenet

# Embedding + DINO counts + bbox features
python model-training/eval/linear_probe.py --backbone dino_counts --extra-features
```

### Classification ablation
```bash
# Full sweep: imagenet × dino_counts, full / 600 / 400 training examples
python model-training/run_experiments_new.py --epochs 45

# Single condition
python model-training/run_experiments_new.py --backbones dino_counts --n-trains full
```
Outputs → `model-training/brightness-class-runs/`
Summary → `model-training/brightness_class_results.csv`

---

## Checkpoints and outputs

| File | What it is |
|------|-----------|
| `best_efficientnet_multihead.pt` | "dino_counts" backbone (tree/streetlight/storefront counts) |
| `ssl-pretrain/best_ssl_backbone.pt` | SimCLR SSL backbone |
| `brightness-regression-run/best_efficientnet_brightness.pt` | Best multi-target regression model |
| `finetune-runs/<backbone>/n<N>/fold_<k>/best_model.pt` | Per-fold checkpoints from k-fold ablation |
| `brightness-class-runs/<backbone>/n<tag>/best_efficientnet_brightness_class.pt` | Classification checkpoints |

---

## Final results (see paper for full detail)

| Backbone | Test MAE | Test RMSE | Test R² |
|----------|----------|-----------|---------|
| Ridge regression (DINO counts, no fine-tuning) | 0.779 | 0.955 | 0.081 |
| ImageNet | 0.486 ± 0.009 | 0.634 ± 0.006 | **0.460 ± 0.011** |
| SSL | 0.498 ± 0.017 | 0.648 ± 0.020 | 0.436 ± 0.035 |
| DINO-counts | 0.509 ± 0.016 | 0.667 ± 0.022 | 0.402 ± 0.040 |
| DINO-counts + warmup | 0.501 ± 0.019 | 0.655 ± 0.020 | 0.417 ± 0.025 |
| DINO-counts v2 + warmup (bbox) | 0.513 ± 0.019 | 0.669 ± 0.018 | 0.399 ± 0.032 |