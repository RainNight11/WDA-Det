# 当前模型方案说明（与代码一致）

本文档描述当前仓库中正在使用的 WDA 一致性训练方案，内容与以下实现保持一致：
- `models/wda_consistency_model.py`
- `networks/trainer.py`
- `data/datasets.py`
- `train.py`
- `validate.py`
- `configs/config_train.py`

## 1. 方案定位

当前方案是“**训练期一致性正则 + 推理期单路径去噪**”：
- 推理只走：`x -> wavelet denoise -> xw -> backbone -> classifier`
- 不再使用 residual/token-gating 分支进行推理调制

目标是让模型在输入扰动下保持判别稳定，同时保持推理图像路径简单、可解释。

## 2. 模型结构

`WDAModel`（`models/wda_consistency_model.py`）包含：
- Backbone：`CLIP` 或 `DINOv2`
- 归一化层：`BN/LN/Identity`（默认 `BN`）
- 分类头：`Linear(feat_dim -> 1)`
- 投影头：两层 MLP（用于一致性特征约束）
- 小波去噪模块：
  - `learn_wavelet=False` 时：使用 `pywt + BayesShrink`（当前主用）
  - `learn_wavelet=True` 时：使用可学习软阈值 DWT/IDWT 分支

## 3. 训练损失（当前实现）

`Trainer.get_loss()`（`networks/trainer.py`）中 RFNT 分支的当前逻辑：
- 监督分支：使用干净去噪图 `xw`
  - `pred_clean = model.forward_denoised(xw)`
  - `L_sup = BCEWithLogits(pred_clean, y)`
- 一致性分支：使用两个增强视图 `xw1, xw2`
  - `xw1 = aug(xw)`, `xw2 = aug(xw)`
  - 学生分支提取 `proj1`，教师分支提取 `proj2`
  - `L_cons = 1 - cos(normalize(proj1), normalize(proj2))`
- 总损失：
  - `L = L_sup + w(t) * L_cons`
  - `w(t)` 由 `consistency_lambda` 和 `consistency_warmup` 控制

教师机制：
- `consistency_ema_decay > 0` 时启用 EMA teacher
- 否则 teacher 为 stop-grad 的当前模型

## 4. 数据处理与增强

### 训练数据（`data/datasets.py`）
- 基础流程：`resize -> (可选)data_augment -> crop -> flip -> normalize`
- 训练增强开关已接入：仅当 `opt.isTrain and opt.data_aug` 时启用 `data_augment`
- `data_augment` 包含两类随机扰动：
  - JPEG 压缩：`jpg_prob`, `jpg_qual`, `jpg_method`
  - 高斯模糊：`blur_prob`, `blur_sig`

### 训练内验证（`train.py:get_val_opt`）
- `isTrain=False`
- `no_resize=False`, `no_crop=False`, `no_flip=True`
- `blur_prob=0.0`, `jpg_prob=0.0`
- 即：验证不做随机增强，但与训练/测试保持同一基础预处理路径

### 独立测试（`validate.py`）
- 已支持按配置构建与训练一致的 transform 规则（`build_transform`）
- 默认 `MEAN/STD` 保持本文件内定义

## 5. 当前默认实验配置（节选）

见 `configs/config_train.py` 当前激活预设（`EXPERIMENT_NAME`）：
- `arch="RFNT-CLIP:ViT-L/14"`
- `fix_backbone=True`
- `learn_wavelet=False`（固定小波）
- 训练增强已开启：
  - `data_aug=True`
  - `jpg_prob=0.5`, `jpg_qual=[30,100]`
  - `blur_prob=0.3`, `blur_sig=[0.0,2.0]`
- 一致性参数：
  - `consistency_lambda=0.1`
  - `consistency_warmup=0.1`
  - `consistency_ema_decay=0.99`
  - `consistency_noise_std=0.01`
  - `consistency_blur_prob=0.5`
  - `consistency_resize_scale=0.1`

## 6. 当前方案的已知特性

- 该方案在 in-domain（如 ProGAN 同域）通常较高
- 跨生成器时，`AP` 通常比 `Acc@0.5` 更稳定
- 不同测试集的最佳阈值可能差异较大，说明存在校准漂移
- 若开启 `learn_wavelet`，需谨慎，当前默认仍推荐固定小波

## 7. 运行方式

- 训练：`python train.py`
- 测试：`python validate.py`

建议在实验记录中同时保存：
- `AP`
- `Acc@0.5`
- `Acc@best_thres`
- `best_thres`

## 8. 论文式方法描述（详细版，可直接改写进 Method）

### 8.1 任务定义与符号

给定输入图像 `x in R^(3xHxW)`，模型输出二分类 logit `s`，预测概率 `p = sigma(s)`，标签 `y in {0,1}`。  
我们关注跨生成器场景：训练主要来自单一或少数生成器域，测试来自未知生成器域。目标是提升 out-of-domain 泛化。

符号定义如下：
- `D(.)`：去噪算子（当前默认为固定小波 BayesShrink）。
- `f(.)`：视觉编码器（CLIP/DINOv2 主干）。
- `g(.)`：分类头（线性层）。
- `h(.)`：投影头（两层 MLP，用于一致性学习）。
- `t1(.), t2(.)`：轻量扰动增强函数。
- `lambda(t)`：一致性权重的训练步调度函数（warm-up）。

### 8.2 方法总览

方法由“单路径推理”和“双视图一致性训练”两部分构成。

**推理阶段（Inference-only Single Path）**  
`x -> D(.) -> xw -> f(.) -> g(.) -> s -> p`  
即推理时不使用任何 residual 引导或双分支结构，保证部署路径简单且叙事一致。

**训练阶段（Supervised + Consistency）**  
监督分支始终使用干净去噪图 `xw`；一致性分支在 `xw` 上构造双视图 `xw1, xw2`，约束特征表示在扰动下保持稳定。

### 8.3 去噪模块设计

当前实现支持两种模式，统一通过 `WDAModel.denoise()` 调用：

**模式A：固定小波去噪（默认）**  
- 对每个通道执行 `wavedec2` 多层分解（`db4`, 默认 3 层）。
- 对高频子带 `(cH, cV, cD)` 使用 BayesShrink 阈值：
  - `sigma = median(|cD|) / 0.6745`
  - `sigma_x = sqrt(max(std(cD)^2 - sigma^2, 0))`
  - `theta = sigma^2 / max(sigma_x, eps)`
- 执行软阈值并 `waverec2` 重建得到 `xw`。

**模式B：可学习小波阈值（可选）**  
- 用可微 DWT/IDWT 卷积替代 pywt 离散实现。
- 每层每子带每通道阈值由可学习参数 `wavelet_theta` 给出，经 `softplus` 保证非负。
- 默认不启用，避免早期训练不稳定。

### 8.4 主干与判别头

去噪后图像先按主干统计量归一化，再送入冻结或部分冻结的编码器：
- `z = f(xw)`，其中 `z in R^C`。
- 归一化层 `norm`（BN/LN/Identity）作用于 `z`。
- 分类 logit：`s = g(norm(z))`。

一致性分支额外使用投影头：
- `q = h(z)`，用于构建对比空间而非直接分类空间。
- 这样可降低一致性约束对判别边界的直接干扰。

### 8.5 训练目标与优化

**监督损失（干净视图）**  
`L_sup = BCEWithLogits(g(f(xw)), y)`  
监督仅在干净去噪图上计算，保证决策目标与推理路径一致。

**一致性损失（增强视图）**  
生成双视图：
- `xw1 = t1(xw)`
- `xw2 = t2(xw)`

提取投影特征：
- `q1 = h(f(xw1))`
- `q2 = h(f(xw2))`

采用归一化余弦一致性：
`L_cons = 1 - <normalize(q1), normalize(q2)>`

**总损失**
`L = L_sup + lambda(t) * L_cons`

其中：
- `lambda(t)` 在训练前期由 0 线性升至 `consistency_lambda`，减少初期优化冲突。
- 若启用 EMA teacher，则 `q2` 由教师网络生成；否则使用 stop-grad teacher。

### 8.6 训练算法流程（对应当前实现）

单个迭代可写为：

1. 输入 batch `x, y`。
2. 去噪得到 `xw = D(x)`（`denoise_and_normalize`）。
3. 监督分支前向：`pred_clean = g(f(xw))`，计算 `L_sup`。
4. 构造双视图：`xw1 = aug(xw)`, `xw2 = aug(xw)`。
5. 学生分支输出 `q1`，教师分支输出 `q2`。
6. 计算 `L_cons`。
7. 合成总损失 `L`，反向传播更新学生参数。
8. 若启用 EMA，则更新教师参数：
   `theta_t <- m * theta_t + (1-m) * theta_s`。

### 8.7 增强策略与鲁棒性来源

当前方案有两类增强，作用层次不同：

**输入增强（可开关）**  
在 dataloader 的 `data_augment` 中执行：
- JPEG 压缩（概率 `jpg_prob`，质量 `jpg_qual`）
- 高斯模糊（概率 `blur_prob`，强度 `blur_sig`）

**一致性增强（损失内部）**  
在 trainer 内部执行：
- 随机缩放回采样（`consistency_resize_scale`）
- 随机模糊（`consistency_blur_prob`, `consistency_blur_sigma_*`）
- 小幅噪声（`consistency_noise_std`）

两者配合的设计意图：
- 输入增强提升样本多样性；
- 一致性增强显式约束“同一语义不同扰动”的特征稳定性。

### 8.8 设计原则与理论直觉

本方法强调两个原则：

**原则1：推理结构极简且自洽**  
去噪后直接分类，不再引入 residual 引导分支，避免“去噪后再引噪”的结构矛盾。

**原则2：把鲁棒性学习放到目标函数，而非推理图**  
通过一致性损失学习扰动不变性，代替手工构造复杂注意力路径。

这一设计在跨域检测中的优势是：
- 部署成本低；
- 结构解释性强；
- 能通过 `lambda(t)`、增强强度和 teacher 机制平衡“判别性”与“稳健性”。

### 8.9 实现对应关系（便于论文与代码互证）

- 去噪与前向：`models/wda_consistency_model.py`
- 监督与一致性损失：`networks/trainer.py:get_loss`
- EMA teacher：`networks/trainer.py:_update_ema_teacher`
- 训练增强：`data/datasets.py:data_augment`
- 一致性增强：`networks/trainer.py:_consistency_augment`
- 训练内验证配置：`train.py:get_val_opt`
- 独立测试与阈值评估：`validate.py`

### 8.10 现阶段局限与后续改进方向

**局限1：域外阈值漂移明显**  
跨生成器时 `best_thres` 变化大，`Acc@0.5` 波动显著。

**局限2：单源训练的域偏置**  
若训练只来自 ProGAN，模型可能偏向学习源域特征，对扩散模型泛化不足。

**局限3：固定去噪参数的偏差风险**  
去噪强度对不同生成器并不总是最优，可能压制部分判别细节。

**后续方向（不改变主框架）**
- 以 `AP/AUC` 主导 checkpoint 选择，弱化 `Acc@0.5`。
- 引入 stress-val 做全局阈值校准。
- 使用 worst-case 视图损失（近似 DRO）替代平均一致性。
