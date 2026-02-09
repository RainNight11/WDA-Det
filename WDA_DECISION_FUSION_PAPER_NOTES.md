# WDADecisionFusionModel 论文式解读与消融设计

对应实现：`models/wda_decision_fusion_model.py`（类 `WDADecisionFusionModel`）。

本文档目标：
- 用论文“方法（Method）”的口径解释模型：动机、符号、结构、融合公式与可解释性输出。
- 给出一次前向的“张量尺寸流”（便于画网络结构图、写实现细节）。
- 给出可直接用于论文的消融实验表（每项消融对应一个清晰假设）。

---

## 1. 问题定义与符号

给定输入图像 `x ∈ R^{3×H×W}`，标签 `y ∈ {0,1}`（0=real，1=fake），模型输出二分类 logit `s`，概率 `p = σ(s)`。

记：
- `D(·)`：去噪算子（默认：固定小波 + BayesShrink 风格阈值收缩；可选：可学习 soft-threshold）。
- `f(·)`：预训练视觉编码器（CLIP 或 DINOv2）。
- `g_m(·)`：主分支分类头（线性层）。
- `g_a(·)`：辅助分支全局分类头（线性层）。
- `A(·)`：由残差分支产生的空间证据图（evidence map / attention）。
- `q(·) ∈ [0,1]`：样本级门控（confidence gate）。
- `γ`：全局融合强度（可学习标量，初始化为 0）。

---

## 2. 方法总览（Main + Evidence + Decision Fusion）

### 2.1 去噪与残差构造

输入 `x` 在进入去噪前会被还原到像素域（与 backbone 的 Normalize 统计量一致），得到：

`x_raw = denorm(x)`

去噪后得到 `xw_raw = D(x_raw)`，并重新归一化送入 backbone：

`xw = norm(xw_raw)`

残差在像素域定义为：

`r = x_raw - xw_raw`

直觉：
- `xw` 更偏“稳定的全局语义/结构”，适合做主判别；
- `r` 更偏“去噪抑制掉的局部高频差异”，可能携带生成伪迹/压缩块/局部不一致等证据。

### 2.2 主分支（去噪图像判别）

主分支对 `xw` 做全局表征并输出主 logit：

`z_m = f(xw)`  
`ŝ_m = Norm(z_m)`  
`s_main = g_m(ŝ_m)`

其中 `Norm` 可为 BN/LN/Identity（实现里默认 BN1d）。

### 2.3 证据分支（残差证据学习）

证据分支以残差 `r` 为输入，输出：
- 空间证据图 `A`（可视化与解释用）
- 辅助全局 logit `s_aux`
- 证据对齐的局部向量 `z_a`（用于门控）

一个关键设计是 **evidence-aligned pooling**：

1) CNN 得到残差特征图 `F`（实现里记为 `feat_map`）。  
2) 通过 `1×1 conv + sigmoid` 得到低分辨率证据图 `A_low`。  
3) 用 `A_low` 对 `F` 做加权池化得到局部证据向量：

`z_a = sum_{u,v}( A_low(u,v) · F(u,v) ) / ( sum_{u,v} A_low(u,v) + ε )`

直觉：门控不只看“主干特征”，也看“证据区域聚合后的残差描述”，更利于抑制无关纹理/边缘噪声造成的误导。

### 2.4 决策级门控融合（核心公式）

门控 `q` 由主分支全局特征与证据向量共同预测：

`q = σ( MLP([ŝ_m, z_a]) )`

最终融合在 logit 层进行（decision-level fusion）：

`s_final = s_main + γ · q · tanh(s_aux)`

其中：
- `γ` 初始化为 0，使训练初期近似退化为纯主分支，稳定启动；
- `q ∈ [0,1]` 是样本级开关，避免辅助证据对所有样本强制生效；
- `tanh(s_aux)` 对辅助 logit 做限幅（[-1,1]），减少极端输出对校准和训练稳定性的破坏。

---

## 3. 一次前向的张量尺寸流（用于画结构图）

以下按常用输入 `H=W=224` 写尺寸；批量大小记为 `B`。实现中证据分支的下采样只发生一次（`MaxPool stride=2`）。

### 3.1 输入与去噪

- 输入（已标准化）：`x`: `[B, 3, 224, 224]`
- 反归一化像素域：`x_raw`: `[B, 3, 224, 224]`
- 去噪像素域：`xw_raw`: `[B, 3, 224, 224]`
- 再归一化：`xw`: `[B, 3, 224, 224]`
- 残差：`r = x_raw - xw_raw`: `[B, 3, 224, 224]`

### 3.2 主分支（backbone + head）

- Backbone 特征：`z_m = f(xw)`: `[B, C]`
  - CLIP `ViT-L/14`: `C=768`
  - DINOv2 `ViT-L14`: `C=1024`（其它配置见文件头 `CHANNELS`）
- 归一化：`ŝ_m = Norm(z_m)`: `[B, C]`
- 主 logit：`s_main = Linear(ŝ_m)`: `[B, 1]`
- 投影头（用于一致性正则时的特征约束）：`proj = Projector(z_m)`: `[B, P]`
  - `P = min(256, C)`

### 3.3 证据分支（residual CNN）

记残差特征图为 `F`。

- `aux_stem`: `Conv3×3(3→32, s=1) + BN + ReLU`
  - 输出：`[B, 32, 224, 224]`
- `aux_pool0`: `MaxPool3×3, stride=2`
  - 输出：`[B, 32, 112, 112]`
- `aux_proj`: `Conv3×3(32→64, s=1) + BN + ReLU`
  - 输出：`F = [B, 64, 112, 112]`
- `ResidualBlock ×2`（通道不变）
  - 输出：`F = [B, 64, 112, 112]`

证据图与辅助 logit：
- `evidence_head`: `Conv1×1(64→1)` 得到 `evidence_logits`: `[B, 1, 112, 112]`
- `A_low = sigmoid(evidence_logits)`: `[B, 1, 112, 112]`
- `A = upsample(A_low → 224×224)`: `[B, 1, 224, 224]`
- 全局池化：`GAP(F)`: `[B, 64, 1, 1] → aux_global: [B, 64]`
- `s_aux = Linear(aux_global)`: `[B, 1]`

证据对齐向量：
- `weighted_sum = sum(F * A_low over spatial)`: `[B, 64]`
- `weight_norm = sum(A_low over spatial)`: `[B, 1]`
- `z_a = weighted_sum / (weight_norm + ε)`: `[B, 64]`

门控与融合：
- `gate_input = concat([ŝ_m, z_a])`: `[B, C+64]`
- `q = aux_gate(gate_input)`: `[B, 1]`
- `s_final = s_main + γ · q · tanh(s_aux)`: `[B, 1]`

### 3.4 输出（论文里怎么描述）

模型至少输出：
- `s_final`: 最终判别 logit（用于监督与推理）
- `A`: 空间证据图（用于解释/可视化）
- `q`: 样本级门控（用于分析“辅助分支何时生效”）
- `s_main, s_aux`: 两个分支 logit（用于定量拆解贡献）

实现中通过 `return_feature` / `return_attention` 开关可取回这些中间量。

---

## 4. 训练目标（写在论文里需要“自洽”）

在当前训练逻辑中（见 `networks/trainer.py` 的 `RFNTDF-*` 分支），监督是直接作用在融合 logit 上：

`L_sup = BCEWithLogits(s_final, y)`

如果同时启用一致性正则（可选项），则对去噪图像 `xw` 做轻量扰动得到两视图 `xw1, xw2`，用 projector 特征做余弦一致性：

`L_cons = 1 - cos( normalize(proj1), normalize(proj2) )`

总损失：

`L = L_sup + λ(t) · L_cons`

这段写法的关键点是：**监督与最终推理路径一致（都依赖 s_final）**，一致性只约束表征稳定性，不改变推理时的输入形式。

---

## 5. 消融实验表（建议直接放进论文）

表中的每一项都对应一个“单一假设”，避免消融解释时混淆因果。

| 消融编号 | 变体名称 | 改动（相对完整模型） | 主要验证的假设 | 常见预期现象（经验） |
|---|---|---|---|---|
| A0 | Main-only | 令 `γ=0`（或不加 residual 分支） | 主分支去噪判别是稳健基线 | 性能回落但更稳定；可作为强基线 |
| A1 | No-gate | 固定 `q=1`（去掉门控 MLP） | 门控能抑制“无关 residual”误导 | 若 residual 易学到边缘/纹理，A1 易掉点、阈值漂移更大 |
| A2 | No-tanh | 去掉 `tanh`，直接 `s_main + γ·q·s_aux` | 限幅对训练稳定与校准有益 | 容易出现 logit 爆炸/过校准，Acc@0.5 波动增大 |
| A3 | Fixed-γ | 固定 `γ=1`（不可学习且非零初始化） | `γ` 的“从 0 启动”是否关键 | 训练早期更不稳，辅助分支可能抢梯度 |
| A4 | Supervise-main | 监督改为 `BCE(s_main,y)`（不监督融合） | 监督对齐 `s_final` 是否必要 | 往往导致融合贡献不明确或 `γ` 学不起来 |
| A5 | No-attn-pool | 令 `z_a = GAP(F)`（不使用 `A_low` 加权池化） | evidence-aligned pooling 是否关键 | 门控更难聚焦证据区域，`q` 解释性下降 |
| A6 | No-attn-map | 用常数图替代 `A_low`（或不学 `evidence_head`） | 显式空间证据是否必要 | 可视化退化，局部伪迹定位能力下降 |
| A7 | Residual-inject (特征融合) | 将 residual 特征注入主分支特征/输入（非本实现） | “特征级融合”是否优于“决策级融合” | 常见更不稳且可解释性差；用于对比决策级的优势 |
| A8 | No-denoise | 令 `xw=x`（不去噪）但仍算 residual | 去噪是否提供关键的“参照系” | residual 变弱或变噪，证据分支学到无关高频更明显 |
| A9 | Learnable-wavelet | `learn_wavelet=True` | 可学习阈值能否适配数据域 | 需谨慎；可能掉点或不稳定（取决于训练策略） |

论文写作建议：
- 最少呈现：A0/A1/A2/A5（覆盖“分支是否必要、门控是否必要、限幅是否必要、证据对齐是否必要”）。
- 若篇幅允许，再补 A3/A4/A8（解释训练稳定性与推理自洽性）。

---

## 6. 可视化与分析（论文图建议）

建议至少做三类分析图：

1) **证据图热力图**：展示 `A` 叠加到原图上（real vs fake 对比）。  
关注点：`A` 是否在伪迹区域更集中，是否跨生成器保持一致模式。

2) **门控分布**：统计不同数据集/生成器下的 `q` 分布（箱线图/直方图）。  
关注点：`q` 是否在“难样本/伪迹明显样本”上更高。

3) **分支贡献分解**：对比 `s_main` 与 `γ·q·tanh(s_aux)` 的幅度与符号。  
关注点：辅助分支更像“纠偏项”还是“放大项”，是否造成阈值漂移。

---

## 7. 写作要点（把实现差异讲清楚）

在论文里建议强调以下“设计取舍”：
- **不做 residual 注入主分支**：避免“去噪后再引入噪声”的结构矛盾；把 residual 作为独立证据分支处理。
- **决策级融合而非特征级融合**：减少对 backbone 表征空间的干扰，训练更稳定，解释更直接。
- **门控 + 限幅 + γ=0 启动**：三者共同保证辅助分支“可用但不强迫用”，并降低校准风险。

---

## 8. 复现信息（建议在论文实验部分写清）

建议报告（与实现默认一致）：
- backbone 名称（如 `CLIP:ViT-L/14` 或 `DINOv2:ViT-L14`）
- wavelet：`db4`，levels=3（以及是否 `learn_wavelet`）
- 输入分辨率（224）与 Normalize 的均值方差来源
- 训练时是否启用一致性正则（`λ(t)`、warmup、增广强度）
- 评估指标：`AP`、`Acc@0.5`、`Acc@best`、`best_thres`（跨域时阈值漂移尤其要报告）

