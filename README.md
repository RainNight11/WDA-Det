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
- `ablation/`: ablation scripts
  - `train_module_ablation.py` / `test_module_ablation.py`: module and loss ablations
  - `train_other_wavelets.py` / `test_other_wavelets.py`: wavelet-family ablations
  - `summarize_ablations.py`: table/CSV aggregation
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

## 5. Ablation Experiments

### 5.1 Module + Loss Ablations
Default behavior:
- 5 epochs per run
- validation on `last` epoch
- already-finished runs are skipped automatically

```bash
# Module ablation: Full / w.o Denoise / w.o Residual / w.o Gating
python ablation/train_module_ablation.py --suite module
python ablation/test_module_ablation.py --suite module --epoch last

# Loss ablation: Full / w.o L_sup / w.o L_cons
python ablation/train_module_ablation.py --suite loss
python ablation/test_module_ablation.py --suite loss --epoch last
```

Force retraining:
```bash
python ablation/train_module_ablation.py --suite module --no-skip-trained
```

### 5.2 Wavelet Family Ablation
```bash
python ablation/train_other_wavelets.py
python ablation/test_other_wavelets.py --val-epoch last
```

Default wavelets in ablation script:
- `coif2`
- `bior4.4`
- `sym8`

(`db4` is used as the main baseline.)

### 5.3 Summarize Results
```bash
python ablation/summarize_ablations.py
```

Generated files in `ablation/`:
- `module_ablation_summary.csv`
- `module_ablation_table.md`
- `loss_ablation_summary.csv`
- `loss_ablation_table.md`
- `ablation_summary_all.csv`
- `ablation_table_all.md`

## 6. Frequently Edited Settings
- Training batch size: `configs/config_train.py` (`batch_size` in selected preset)
- Validation batch size: `configs/config_validate.py` (`batch_size` in selected preset)
- Training epochs: `configs/config_train.py` (`niter`) or script arg `--niter`
- Validation checkpoint selection: `configs/config_validate.py` (`val_epoch`, supports `last`)

## 7. Notes
For method-level details, see:
- `WDA_DECISION_FUSION_PAPER_NOTES.md`
