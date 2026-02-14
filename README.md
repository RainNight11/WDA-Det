# WDA-Det: Wavelet Denoising with Decision-Level Fusion for Real/Fake Image Detection

Official PyTorch implementation of our paper-style real-vs-fake image detector.

WDA-Det combines:
- a denoised semantic branch (`x -> denoise -> backbone -> main logit`),
- a residual evidence branch (`x_raw - x_denoised -> CNN -> evidence + aux logit`),
- and a gated decision-level fusion (`main + gamma * gate * tanh(aux)`).

The main model lives in `models/wda_decision_fusion_model.py`.

## Highlights
- Dual-view decision design: semantic decision from denoised input + forensic evidence from residuals.
- Gated fusion for robust cross-domain behavior.
- Support for CLIP and DINOv2 backbones.
- Unified training/evaluation pipeline with CSV-based reporting.

## Repository Structure
- `train.py`: training entry point.
- `validate.py`: offline multi-dataset evaluation.
- `test.py`, `main.py`: ad-hoc inference scripts.
- `models/`: model definitions (`wda_decision_fusion_model.py`, `wda_consistency_model.py`).
- `networks/trainer.py`: optimization and loss logic.
- `data/datasets.py`: dataset loading and augmentations.
- `configs/config_train.py`: train experiment presets.
- `configs/config_validate.py`: validation presets.
- `dataset_paths.py`: per-benchmark sub-dataset lists.
- `checkpoints/`, `TestResults/`, `ResultsAnalysis/`: outputs and analysis artifacts.

## Method Overview
Primary variant: `RFNTDF-*` (decision fusion)

1) Denoising branch
- Convert normalized tensor back to pixel space.
- Apply wavelet denoising (`pywt` by default).
- Re-normalize and feed backbone for `s_main`.

2) Evidence branch
- Build residual evidence `r = x_raw - xw_raw`.
- Use lightweight CNN to predict evidence map and auxiliary logit `s_aux`.
- Pool local evidence feature for gating.

3) Decision fusion
- Gate `q in [0,1]` is predicted from `[z_main, z_local]`.
- Final logit: `s_final = s_main + gamma * q * tanh(s_aux)`.

Detailed notes: `WDA_DECISION_FUSION_PAPER_NOTES.md`.

## Environment
Install dependencies according to `README/readme_package_install.txt`.

Typical packages:
- `torch`, `torchvision`
- `pywavelets`
- `scikit-learn`
- `opencv-python`
- `tensorboard`

## Dataset Conventions
Training loader: `data/RealFakeDataset` in `data/datasets.py`.

Expected naming convention relies on path keywords:
- real images contain `0_real`
- fake images contain `1_fake`

### WildRF (`data_mode="WildRF"`)
`wang2020_data_path` should contain:
- `train/` (`...0_real...`, `...1_fake...`)
- `val/` (`...0_real...`, `...1_fake...`)

### ProGAN-style (`data_mode="wang2020"`)
`wang2020_data_path` should contain:
- `train/progan/`
- `test/progan/`

### Offline validation benchmarks
`validate.py` reads sub-dataset names from `dataset_paths.py`:
- `fdmas`: `ADM`, `biggan`, `cyclegan`, `DALLE2`, ...
- `WildRF`: `facebook`, `reddit`, `twitter`

## Quick Start

### 1) Select a training preset
Edit `configs/config_train.py`:

```python
EXPERIMENT_NAME = "wda_decision_fusion_v1_WildRF"
```

Available paper-relevant presets include:
- `wda_decision_fusion_v1_WildRF` (recommended)
- `wda_decision_fusion_v1_fdmas` (recommended)
- `wda_consistency_v1_WildRF` (baseline)
- `wda_consistency_v1_fdmas` (baseline)

### 2) Train
```bash
python train.py
```

Outputs:
- checkpoint: `checkpoints/<data_name>/<exp_name>/model_epoch_*.pth`
- log: `checkpoints/<data_name>/<exp_name>/training.log`

TensorBoard:
```bash
tensorboard --logdir checkpoints --port 6006
```

### 3) Validate
Edit `configs/config_validate.py`:

```python
VAL_EXPERIMENT_NAME = "wda_decision_fusion_v1_WildRF"
```

Run:
```bash
python validate.py
```

Outputs:
- `TestResults/.../<VAL_EXPERIMENT_NAME>/epoch*/epoch_*_results.csv`
- `validate.log`

## Evaluation Metrics
`validate.py` reports:
- `AP`
- `Acc(0.5)`
- `Acc(best)`
- `best_thres`

For cross-domain reporting, we recommend presenting all of the above, especially `Acc(best)` and `best_thres` to quantify threshold shift.

## Model Selection (`opt.arch`)
- Decision fusion (primary): `RFNTDF-CLIP:ViT-L/14`, `RFNTDF-DINOv2:ViT-L14`, ...
- Consistency baseline: `RFNT-CLIP:ViT-L/14`, `RFNT-DINOv2:ViT-L14`, ...

Routing entry: `models/__init__.py`.

## Reproducibility Checklist
When reporting paper results, record:
- backbone family and variant,
- wavelet settings (`db4`, levels, learnable or fixed),
- normalization statistics source (CLIP vs DINOv2),
- consistency settings (`consistency_*`),
- evaluation protocol and threshold policy.

## Citation
If you use this code, please cite our paper.

```bibtex
@article{wdadet2026,
  title   = {WDA-Det: Wavelet Denoising with Decision-Level Fusion for Real/Fake Image Detection},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2026}
}
```

You can replace this entry with the final publication metadata later.

## Acknowledgment
- CLIP backbone and tooling from OpenAI CLIP ecosystem.
- DINOv2 backbone support for strong visual representations.
