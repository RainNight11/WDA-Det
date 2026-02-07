# WDA 方案与尝试过程记录

本记录用于梳理当前 WDA（去噪+一致性）方案的最终形态，以及从早期残差/注意力分支到一致性正则的尝试过程，便于复现实验与撰写论文方法部分。

## 1. 当前最终方案（已落地）

**核心目标**：推理阶段只走“去噪单路径”，训练阶段通过一致性正则提升对部署扰动的鲁棒性。

**推理路径（单路径）**  
`x -> wavelet denoise -> xw -> backbone f -> head g -> y`

**训练路径（监督 + 一致性）**
- 监督损失：使用**干净 xw**  
  `L_sup = BCE(y, g(f(xw)))`
- 一致性损失：只使用**增强视图 xw1 / xw2**  
  `xw1 = t1(xw), xw2 = t2(xw)`  
  `L_cons = 1 - cos(norm(z1), norm(z2))`
- 总损失：  
  `L = L_sup + λ(t) * L_cons`，`λ(t)` 采用 warm-up

**Teacher/Student**  
使用 stop-grad teacher；可选 EMA teacher（在 `networks/trainer.py` 里实现）。

**增强（轻量）**  
轻微 resize / blur / noise（对应 `consistency_*` 参数），避免强颜色/裁剪/旋转。

**实现位置**
- 模型：`models/wda_consistency_model.py`
- 训练逻辑：`networks/trainer.py`
- 训练配置：`configs/config_train.py`

## 2. 小波去噪策略（当前状态）

### 已恢复到“不可学习小波”
- 当 `learn_wavelet=False` 时，`denoise()` 使用 **pywt + BayesShrink**（逐图像自适应阈值）。  
- 当 `learn_wavelet=True` 时，走 **可学习 soft-threshold**（DWT/IDWT 卷积实现）。

> 结论：**目前不可学习模式与旧版一致**，避免了可学习小波导致的性能下降。

## 3. 方案演进与关键尝试

### 3.1 残差注意力 / Token Gating 路线
**初衷**：残差信号引导模型关注伪迹区域。  
**问题**：
- residual 可能强化边缘/压缩块，而非伪迹；
- token 对齐偏差导致 gating 学偏；
- 在 ViT 上效果不稳定。

**尝试改进**：
- 改“残差能量”到“小维度 residual embedding”（Conv->Pool）；
- 添加 token 平滑；
- 改 BN/LN/Identity；
**结果**：整体收益有限，鲁棒性波动大。

### 3.2 一致性正则路线（当前方案）
**动机**：不强行定位 patch，而是让“去噪视图”对部署扰动保持稳定。  
**优点**：
- 推理图更干净（无 residual 引入）；
- 更稳定、更易解释；
- 对 JPEG/blur 等扰动更直接有效。

**一致性形式演进**：
1. 特征一致性（cosine）——当前主方案  
2. Logit 一致性（MSE/KL）——尝试过，稳定性稍弱  
3. 使用原图 x 作为 teacher ——易把真实纹理当 nuisance 拉平，放弃

### 3.3 EMA Teacher 引入
**目的**：平滑 teacher 输出、稳定一致性目标。  
**现象**：有时训练更稳，但测试不一定提升；需调 `consistency_ema_decay` 与 λ。

### 3.4 可学习小波尝试
**实现**：DWT/IDWT 卷积 + soft-threshold（`wavelet_theta`）  
**现象**：精度下降明显（≈10pt）。  
**结论**：保留“不可学习小波”作为最终默认。

## 4. 训练配置（当前推荐）

在 `configs/config_train.py` 的 `wda_consistency_v1` 中：
- `learn_wavelet=False`（固定小波）
- `consistency_lambda` 适中（如 0.1~0.2）
- `consistency_warmup=0.1`
- `consistency_ema_decay` 可开可关（0.9~0.99 需试）
- 轻量增强参数：
  - `consistency_noise_std` 小于 0.01
  - `consistency_blur_prob` 0.3~0.5
  - `consistency_blur_sigma_min/max` 0.5~1.2
  - `consistency_resize_scale` 0.05~0.1

## 5. 当前结论（写在论文里的关键点）

- 推理阶段**只用去噪图**，完全消除“去噪又引噪”的逻辑矛盾。  
- 训练阶段通过**去噪视图的一致性正则**增强鲁棒性，不依赖 residual 对齐。  
- 与残差 gating 相比，该方案更稳定、更易解释、对部署扰动提升更直接。

## 6. 后续建议（可选）

- 若 EMA 影响测试，可调低 `consistency_ema_decay` 或临时关闭。
- 若一致性过强导致过平滑，减小 `consistency_lambda` 或缩短 warm-up。
- 需要进一步提升时，可尝试只在 projector 上做一致性，backbone 冻结更稳。
