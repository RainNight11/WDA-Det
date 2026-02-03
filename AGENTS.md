# Repository Guidelines

Use this guide to keep contributions to the ICME ImageDetect repository consistent and reproducible.

## Project Structure & Module Organization
- `train.py`, `validate.py`, `test.py`, `main.py`: primary entry points for training and evaluation runs.
- `configs/`: experiment presets (e.g., `config_train.py`, `config_validate.py`) that select dataset paths, checkpoints, and hyperparameters.
- `options/`: CLI argument parsing for train/test scripts.
- `models/` and `networks/`: model definitions, backbones, and training wrappers.
- `data/`: dataset loading utilities and transforms.
- `ResultsAnalysis/`, `TestResults/`: generated metrics, logs, and evaluation outputs.
- `dataset_paths.py`, `dataset_paths_wang.py`: shared dataset root mappings.
- `vis/`, `visA.py`: visualization utilities and scripts.

## Build, Test, and Development Commands
- Python-only project; install dependencies per `README/readme_package_install.txt` (CUDA-specific Torch versions are listed there).
- `python train.py` runs training using the preset in `configs/config_train.py` (`EXPERIMENT_NAME`).
- `python validate.py` evaluates checkpoints from `configs/config_validate.py` (`VAL_EXPERIMENT_NAME`) and writes into `TestResults/`.
- `python test.py` or `python main.py` runs ad-hoc evaluation/inference using `options/test_options.py` flags.
- `tensorboard --logdir checkpoints --port 6006` streams training metrics.
- `command.txt` contains an example CLI invocation for a configured experiment.

## Coding Style & Naming Conventions
- Follow existing Python conventions: 4-space indentation and `snake_case` for functions/variables.
- Use `CamelCase` for classes and `UPPER_SNAKE_CASE` for constants (e.g., `EXPERIMENT_NAME`).
- Keep configuration keys aligned with existing presets in `EXPERIMENT_CONFIGS` and `VAL_EXPERIMENT_CONFIGS`.
- No formatter/linter is configured; keep imports tidy and avoid unused symbols.

## Testing Guidelines
- No dedicated unit-test framework is present; validation is done via `validate.py` and `test.py`.
- Update dataset roots in `configs/config_validate.py` or `dataset_paths*.py` before running evaluations.
- Record results under `TestResults/` or `ResultsAnalysis/` as appropriate.

## Commit & Pull Request Guidelines
- No Git history is available in this checkout, so no enforced commit convention is known.
- Use short, descriptive messages (e.g., `train: add WildRF preset`, `fix: dataset path mapping`).
- PRs should include: a concise summary, the configs/datasets touched, and metrics or logs for training/validation changes.
- Avoid committing large artifacts (datasets, checkpoints, or TensorBoard logs); keep them external.

## Configuration & Data Tips
- Set experiment switches in `configs/config_train.py` and `configs/config_validate.py` rather than editing scripts.
- Dataset roots typically live outside the repo; keep paths consistent with `dataroot` or `*_data_path` keys.
