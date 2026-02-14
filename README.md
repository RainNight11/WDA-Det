# WDA-Det

WDA-Det is a PyTorch project for real/fake image detection with a dual-branch design:
- Main branch: semantic prediction on wavelet-denoised input
- Evidence branch: local forgery cues from residual signals
- Decision fusion: `s_final = s_main + gamma * q * tanh(s_aux)`

## 1. Repository Structure
- `train.py`: training entry point
- `validate.py`: validation entry point (writes CSV metrics)
- `configs/config_train.py`: training presets (experiment, epochs, batch size, data path)
- `configs/config_validate.py`: validation presets (checkpoint dir, epoch selector, output dir)
- `models/wda_decision_fusion_model.py`: WDA-Det model definition
- `networks/trainer.py`: optimization and loss logic
- `checkpoints/`: model checkpoints and training logs
- `TestResults/`: validation outputs

## 2. Installation
Install a CUDA-compatible `torch/torchvision` first, then install the rest:

```bash
pip install -r requirements.txt
```

## 3. Data Setup
Set dataset paths in configs:
- Training: `configs/config_train.py` (`wang2020_data_path` in your selected preset)
- Validation: `configs/config_validate.py` (`dataroot` in your selected preset)

Naming convention expected by loaders:
- real images contain `0_real`
- fake images contain `1_fake`

## 4. Basic Usage

### 4.1 Train
Recommended main preset: `wda_decision_fusion_v1_WildRF`

```bash
python train.py
```

Outputs:
- checkpoints: `checkpoints/<data_name>/<exp_name>/model_epoch_*.pth`
- log: `checkpoints/<data_name>/<exp_name>/training.log`

### 4.2 Validate
```bash
python validate.py
```

Outputs:
- `TestResults/.../<VAL_EXPERIMENT_NAME>/epoch*/epoch_*_results.csv`
- `validate.log`

Metrics:
- `AP`
- `Acc(0.5)`
- `Acc(best)`
- `best_thres`

## 5. Frequently Edited Settings
- Training batch size: `configs/config_train.py` (`batch_size` in selected preset)
- Validation batch size: `configs/config_validate.py` (`batch_size` in selected preset)
- Training epochs: `configs/config_train.py` (`niter`) or script arg `--niter`
- Validation checkpoint selection: `configs/config_validate.py` (`val_epoch`, supports `last`)

