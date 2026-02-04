# ICME ImageDetect

基于 PyTorch 的图像真伪检测项目，当前默认训练使用 **WDA（去噪 + token-wise gating）** 模型：  
输入图像先做小波去噪，残差信号仅作为特征域的 token gate，不再回注输入域。

## 目录结构
- `train.py` / `validate.py` / `test.py` / `main.py`：训练与评估入口
- `configs/`：训练/验证配置（如 `config_train.py`、`config_validate.py`）
- `models/`：模型实现（`models/wda_model.py` 为当前主模型）
- `networks/`：训练封装、优化器等
- `data/`：数据集加载与增强
- `TestResults/`、`ResultsAnalysis/`：实验输出

## 环境依赖
参考 `README/readme_package_install.txt` 安装依赖（包含 CUDA 版 PyTorch）。  
核心依赖：`torch`、`torchvision`、`pywt`、`tensorboardX`、`scikit-learn`、`opencv-python` 等。

## 数据准备
在 `configs/config_train.py` 的实验配置里设置数据路径，例如：
```python
wang2020_data_path="/your/dataset/path"
```
并确保目录结构与 `data/datasets.py` 中的读取逻辑一致（`0_real`/`1_fake`）。

## 训练
1) 选择实验：在 `configs/config_train.py` 设置  
```python
EXPERIMENT_NAME = "token_gate_v1"
```
2) 运行训练：
```bash
python train.py
```

训练输出：  
`checkpoints/<data_name>/<experiment_name>/model_epoch_*.pth`  
日志：`checkpoints/<data_name>/<experiment_name>/training.log`

TensorBoard：
```bash
tensorboard --logdir checkpoints --port 6006
```

## 验证 / 测试
在 `configs/config_validate.py` 中设置 `VAL_EXPERIMENT_NAME` 与数据路径后运行：
```bash
python validate.py
```

## 重要说明（WDA 模型）
- 仅 **CLIP-ViT** backbone 启用 token gating（如 `RFNT-CLIP:ViT-L/14`）。  
  其他 backbone 会回退到全局特征路径。
- 输入建议为**方形图像**（裁剪后最好是 224×224），否则 token 网格可能不匹配。
- `pywt` 为必要依赖，用于小波去噪。

## 训练配置建议（WDA）
建议配置：
- `loss="loss_bce"`
- `fix_backbone=True`（只训练 gate + head）
- `lr=1e-3`，`batch_size=16/32`

如需扩展（DINOv2 token gating / 额外消融实验），欢迎继续补充。
