# WDA-Det (PyTorch)

WDA-Det is a PyTorch project for **real vs. fake image detection**.

The primary model in this repo is **WDA-Det (Decision Fusion)** implemented in `models/wda_decision_fusion_model.py`:
- Main branch: wavelet denoise → backbone → main head
- Evidence branch: residual (raw − denoised) → CNN → evidence map + auxiliary logit
- Decision-level fusion: gated auxiliary contribution added to the main logit

An older/alternative baseline (kept for comparison) is the **consistency single-path** model in `models/wda_consistency_model.py`.

Entry points:
- Training: `train.py`
- Offline evaluation: `validate.py`
- Ad-hoc inference scripts: `test.py`, `main.py`

Paper-oriented notes for the decision-fusion model:
- `WDA_DECISION_FUSION_PAPER_NOTES.md`

---

## Project Layout

- `train.py`, `validate.py`, `test.py`, `main.py`: runnable scripts
- `configs/`: experiment presets (`config_train.py`, `config_validate.py`)
- `options/`: CLI options
- `models/`: model implementations
- `networks/`: training wrapper (`networks/trainer.py`)
- `data/`: training dataset loader and augmentations (`data/datasets.py`)
- `checkpoints/`: saved checkpoints and logs
- `TestResults/`, `ResultsAnalysis/`: evaluation outputs and analysis artifacts

---

## Requirements

Install dependencies as described in `README/readme_package_install.txt` (CUDA-specific Torch versions may be listed there).

Common dependencies used in this repo include:
- `torch`, `torchvision`
- `pywt` (PyWavelets)
- `scikit-learn`
- `opencv-python`
- `tensorboard` (or TensorBoard-compatible SummaryWriter)

Notes:
- The default wavelet denoiser uses `pywt` and runs via NumPy/CPU in the provided implementation.
- Decision-fusion model selection uses the `RFNTDF-*` prefix (see `models/__init__.py`).

---

## Datasets and Folder Conventions

### Training (`data/datasets.py`)

Training uses `data/RealFakeDataset` and relies on `opt.data_mode` and `opt.wang2020_data_path`.

Typical conventions:

1) **WildRF** (`opt.data_mode="WildRF"`)

`opt.wang2020_data_path` should contain:
- `train/` with images whose paths include `0_real` and `1_fake`
- `val/` with images whose paths include `0_real` and `1_fake`

2) **wang2020 / ProGAN-style** (`opt.data_mode="wang2020"`)

`opt.wang2020_data_path` should contain:
- `train/progan/` with `0_real` and `1_fake`
- `test/progan/` with `0_real` and `1_fake`

The code filters files by checking whether `0_real` / `1_fake` occurs in the file path.

### Offline evaluation (`validate.py`)

`validate.py` evaluates multiple sub-datasets for `fdmas` and `WildRF` using the lists in `dataset_paths.py`:
- `fdmas`: `ADM`, `biggan`, `cyclegan`, `DALLE2`, ...
- `WildRF`: `facebook`, `reddit`, `twitter`

For `WildRF`, your validation `dataroot` is typically:

`/path/to/WildRF/test/`

with subfolders:
- `/path/to/WildRF/test/facebook/...0_real...` and `...1_fake...`
- `/path/to/WildRF/test/reddit/...0_real...` and `...1_fake...`
- `/path/to/WildRF/test/twitter/...0_real...` and `...1_fake...`

---

## Training

Training is driven by `configs/config_train.py`.

1) Select an experiment preset by editing:

`configs/config_train.py` → `EXPERIMENT_NAME = "..."`

Relevant presets in this repo include:
- `wda_decision_fusion_v1_WildRF` (decision-fusion, recommended)
- `wda_decision_fusion_v1_fdmas` (decision-fusion, recommended)
- `wda_consistency_v1_WildRF` (baseline)
- `wda_consistency_v1_fdmas` (baseline)

2) Run:

```bash
python train.py
```

Outputs:
- Checkpoints: `checkpoints/<data_name>/<experiment_name>/model_epoch_*.pth`
- Log: `checkpoints/<data_name>/<experiment_name>/training.log`

TensorBoard:
```bash
tensorboard --logdir checkpoints --port 6006
```

---

## Validation / Testing (Offline, Multi-Dataset)

Validation is driven by `configs/config_validate.py`.

1) Select a validation preset by editing:

`configs/config_validate.py` → `VAL_EXPERIMENT_NAME = "..."`

2) Run:

```bash
python validate.py
```

Outputs are written under:
- `TestResults/<...>/<VAL_EXPERIMENT_NAME>/epoch<k>/epoch_<k>_results.csv`
- plus a `validate.log` in the same folder.

Metrics:
- `AP`
- `Acc(0.5)`
- `Acc(best)` and `best_thres` (helps diagnose threshold drift across domains)

---

## Inference Scripts (`test.py`, `main.py`)

This repo also includes standalone inference-style scripts intended for “folder of images → CSV predictions” workflows:
- `test.py`
- `main.py`

These scripts use `options/test_options.py` and load a checkpoint from `--premodel_path`.

Important:
- `options/test_options.py` defaults `--arch` to `RFNTDF-CLIP:ViT-L/14` (decision fusion).
- The decision-fusion model forward returns a tuple by default: `(logit, feature)`. If a script assumes a single tensor, adjust it to unpack the tuple.

---

## Selecting the Model Variant

Model selection is string-based via `opt.arch`:
- Decision fusion model (primary): `RFNTDF-CLIP:ViT-L/14`, `RFNTDF-DINOv2:ViT-L14`, ...
- Consistency baseline: `RFNT-CLIP:ViT-L/14`, `RFNT-DINOv2:ViT-L14`, ...

Implementation entry:
- `models/get_model()` in `models/__init__.py`

---

## Reproducibility Notes

If you are writing a paper, report:
- Backbone name (CLIP vs DINOv2; exact variant)
- Wavelet settings (`db4`, levels=3; whether `learn_wavelet` is enabled)
- Data normalization stats source (CLIP vs DINOv2)
- Whether consistency regularization is enabled (`consistency_*` options)
- Multi-dataset evaluation protocol and threshold selection (`Acc@0.5` vs `Acc@best`)

---

## Quick Pointers

- Decision-fusion model code (primary): `models/wda_decision_fusion_model.py`
- Consistency baseline code: `models/wda_consistency_model.py`
- Training wrapper / losses: `networks/trainer.py`
- Training dataset loader: `data/datasets.py`
- Offline evaluation runner: `validate.py`
