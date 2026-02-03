#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 GaussianNoiseSimilarity（高斯噪声相似度）合并表的脚本
输入列：dataset, parameter, noise_std, image_type, count, mean_similarity, std_similarity, source_csv
输出：analysis_outputs/ 目录下的汇总 CSV + 多张 PNG 图
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple

# -----------------------------
# 基础配置
# -----------------------------
INPUT_BASENAME = "GaussianNoiseSimilarity"  # 允许带或不带.csv
OUTDIR = "analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10

# -----------------------------
# 工具函数
# -----------------------------
def load_csv_smart(basename: str) -> pd.DataFrame:
    """兼容 'GaussianNoiseSimilarity' 和 'GaussianNoiseSimilarity.csv' 两种写法"""
    candidates = [basename, basename + ".csv"] if not basename.lower().endswith(".csv") else [basename]
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            return df
    raise FileNotFoundError(f"找不到输入文件：{candidates}")

def welch_t_from_summary(m1, s1, n1, m2, s2, n2) -> Tuple[float, float]:
    """基于均值/标准差/样本量计算 Welch t 的统计量与自由度（p 值在外部按需要算/或仅输出 t 与 df）"""
    # t = (m1 - m2) / sqrt(s1^2/n1 + s2^2/n2)
    # df 近似 = (v)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))
    if min(n1, n2) <= 1:
        return np.nan, np.nan
    v1 = (s1**2) / n1
    v2 = (s2**2) / n2
    denom = math.sqrt(v1 + v2) if (v1 + v2) > 0 else np.nan
    t = (m1 - m2) / denom if denom and denom > 0 else np.nan
    df = ((v1 + v2)**2) / ((v1**2)/(n1 - 1) + (v2**2)/(n2 - 1)) if (n1 > 1 and n2 > 1) else np.nan
    return t, df

def cohens_d_from_summary(m1, s1, n1, m2, s2, n2) -> float:
    """Cohen's d（基于合并标准差的 version）。当方差为 0 或样本不足时返回 NaN。"""
    if min(n1, n2) <= 1:
        return np.nan
    sp2 = ((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2) if (n1 + n2 - 2) > 0 else np.nan
    if not np.isfinite(sp2) or sp2 <= 0:
        return np.nan
    sp = math.sqrt(sp2)
    if sp == 0:
        return np.nan
    return (m1 - m2) / sp

def se_difference(s1, n1, s2, n2):
    """
    逐元素计算两均值差的标准误：
    SE(diff) = sqrt( (s1^2)/n1 + (s2^2)/n2 )
    其中 n1<=0 或 n2<=0 时返回 NaN（避免除零/非法）。
    支持传入 pandas Series / numpy array。
    """
    s1 = pd.to_numeric(s1, errors="coerce")
    s2 = pd.to_numeric(s2, errors="coerce")
    n1 = pd.to_numeric(n1, errors="coerce")
    n2 = pd.to_numeric(n2, errors="coerce")

    # n<=0 置为 NaN，避免除零或负数样本量导致的无意义结果
    n1 = n1.where(n1 > 0, np.nan)
    n2 = n2.where(n2 > 0, np.nan)

    return np.sqrt((s1**2) / n1 + (s2**2) / n2)


def spearman_rho(x, y) -> float:
    """Spearman 相关（手写简版，足够用；也可换 scipy）"""
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    return np.corrcoef(xr, yr)[0, 1]

def find_flip_point(noise_vals, gaps) -> Optional[float]:
    """在线性插值下估计 gap=0 的噪声‘翻转点’，若不存在符号变化返回 None"""
    order = np.argsort(noise_vals)
    nv = np.array(noise_vals)[order]
    gv = np.array(gaps)[order]
    for i in range(len(nv) - 1):
        g1, g2 = gv[i], gv[i + 1]
        if np.isnan(g1) or np.isnan(g2):
            continue
        if g1 == 0:
            return float(nv[i])
        if g1 * g2 < 0:
            # 线性插值：g = g1 + (g2-g1)*t, 求 g=0 → t = -g1/(g2-g1)
            t = -g1 / (g2 - g1)
            if 0 <= t <= 1:
                return float(nv[i] + t * (nv[i + 1] - nv[i]))
    return None

# -----------------------------
# 读取与清洗
# -----------------------------
df = load_csv_smart(INPUT_BASENAME)

required_cols = [
    "dataset", "parameter", "noise_std", "image_type",
    "count", "mean_similarity", "std_similarity", "source_csv"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV 缺少列：{missing}")

# 统一类型
df["dataset"] = df["dataset"].astype(str)
df["image_type"] = df["image_type"].astype(str)
df["noise_std"] = pd.to_numeric(df["noise_std"], errors="coerce")
df["count"] = pd.to_numeric(df["count"], errors="coerce")
df["mean_similarity"] = pd.to_numeric(df["mean_similarity"], errors="coerce")
df["std_similarity"] = pd.to_numeric(df["std_similarity"], errors="coerce")

# 基础质检
if df["image_type"].nunique() < 2:
    warnings.warn("警告：image_type 少于两类，缺少 Real/Fake 对比。")

# -----------------------------
# 长表 → 宽表（合并 Real/Fake）
# -----------------------------
# 每个 dataset × noise_std 上，拼出 Real/Fake 两组统计
pivot = df.pivot_table(
    index=["dataset", "noise_std"],
    columns="image_type",
    values=["mean_similarity", "std_similarity", "count"],
    aggfunc="first"
)

# 扁平列名
pivot.columns = [f"{a}_{b}".lower() for a, b in pivot.columns]
pivot = pivot.reset_index()

# 确保 Real/Fake 列存在（没有的补 NaN）
for base in ["mean_similarity", "std_similarity", "count"]:
    for t in ["real", "fake"]:
        col = f"{base}_{t}"
        if col not in pivot.columns:
            pivot[col] = np.nan

# 计算核心派生指标
pivot["gap_mean_real_minus_fake"] = pivot["mean_similarity_real"] - pivot["mean_similarity_fake"]
pivot["se_gap"] = se_difference(
    pivot["std_similarity_real"], pivot["count_real"],
    pivot["std_similarity_fake"], pivot["count_fake"]
)


# Welch t（仅给 t 与 df；p 值如果需要可在外部用 scipy 进一步求，或这里也可近似正态）
t_list, df_list, d_list = [], [], []
for _, r in pivot.iterrows():
    t_stat, dfree = welch_t_from_summary(
        r["mean_similarity_real"], r["std_similarity_real"], r["count_real"],
        r["mean_similarity_fake"], r["std_similarity_fake"], r["count_fake"]
    )
    t_list.append(t_stat)
    df_list.append(dfree)
    d_list.append(cohens_d_from_summary(
        r["mean_similarity_real"], r["std_similarity_real"], r["count_real"],
        r["mean_similarity_fake"], r["std_similarity_fake"], r["count_fake"]
    ))
pivot["welch_t"] = t_list
pivot["welch_df"] = df_list
pivot["cohens_d"] = d_list

# 保存宽表
wide_path = os.path.join(OUTDIR, "GaussianNoiseSimilarity_wide.csv")
pivot.sort_values(["dataset", "noise_std"]).to_csv(wide_path, index=False)
print(f"[OK] 写出：{wide_path}")

# -----------------------------
# 逐数据集：斜率/单调性/翻转点/最大差值点
# -----------------------------
records = []
for ds, g in pivot.groupby("dataset"):
    # 有效点
    sub = g[["noise_std", "gap_mean_real_minus_fake"]].dropna()
    sub = sub.sort_values("noise_std")
    noise_vals = sub["noise_std"].values
    gaps = sub["gap_mean_real_minus_fake"].values

    if len(sub) >= 2:
        # 线性斜率（gap ~ noise）
        slope = np.polyfit(noise_vals, gaps, 1)[0]
        # Spearman 单调性
        rho = spearman_rho(noise_vals, gaps)
        # 翻转点（gap=0）
        flip_std = find_flip_point(noise_vals, gaps)
        # 最大差值点（绝对值 & 符号保留）
        idx_max = int(np.nanargmax(np.abs(gaps))) if len(gaps) else None
        max_gap = float(gaps[idx_max]) if idx_max is not None else np.nan
        max_gap_std = float(noise_vals[idx_max]) if idx_max is not None else np.nan
    else:
        slope = np.nan
        rho = np.nan
        flip_std = None
        max_gap = np.nan
        max_gap_std = np.nan

    records.append({
        "dataset": ds,
        "n_points": len(sub),
        "slope_linear_gap_vs_noise": slope,
        "spearman_rho_gap_noise": rho,
        "flip_noise_std_if_any": flip_std,
        "max_abs_gap": max_gap,
        "max_abs_gap_at_noise_std": max_gap_std
    })

ds_metrics = pd.DataFrame(records).sort_values("dataset")
ds_metrics_path = os.path.join(OUTDIR, "dataset_level_metrics.csv")
ds_metrics.to_csv(ds_metrics_path, index=False)
print(f"[OK] 写出：{ds_metrics_path}")

# -----------------------------
# 可视化
# -----------------------------

def save_tight(fig, path, close=True):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"[FIG] {path}")

# 1) 每个 dataset 的 gap 曲线（含误差带）
def plot_gap_curves(pivot_df: pd.DataFrame):
    ds_list = sorted(pivot_df["dataset"].unique())
    n = len(ds_list)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols, 3.2*rows), squeeze=False)
    for i, ds in enumerate(ds_list):
        ax = axes[i // cols][i % cols]
        sub = pivot_df[pivot_df["dataset"] == ds].sort_values("noise_std")
        x = sub["noise_std"].values
        y = sub["gap_mean_real_minus_fake"].values
        se = sub["se_gap"].values
        ax.plot(x, y, marker="o", linewidth=1.2)
        # 误差带（±1.96*SE 近似 95% CI）
        if np.all(np.isfinite(se)):
            ax.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.15)
        ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_title(ds)
        ax.set_xlabel("Noise STD")
        ax.set_ylabel("Gap (Real - Fake)")
    # 清空空子图
    for j in range(i+1, rows*cols):
        axes[j // cols][j % cols].axis("off")
    save_tight(fig, os.path.join(OUTDIR, "gap_curves_by_dataset.png"))

plot_gap_curves(pivot)

# 2) 森林图：按 noise_std 分面，展示 Cohen's d（含 95% 近似 CI）
def ci_cohens_d_normal_approx(d, n1, n2):
    # 正态近似方差 ≈ (n1+n2)/(n1*n2) + d^2/(2*(n1+n2-2))  （常见近似，足够展示）
    if min(n1, n2) <= 1 or (n1 + n2 - 2) <= 0:
        return (np.nan, np.nan)
    var = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2 - 2))
    se = math.sqrt(var) if var > 0 else np.nan
    return (d - 1.96*se, d + 1.96*se)

def plot_forest_by_noise_std(pivot_df: pd.DataFrame):
    for ns, g in pivot_df.groupby("noise_std"):
        sub = g.copy()
        sub["lo"], sub["hi"] = np.nan, np.nan
        for idx, r in sub.iterrows():
            d = r["cohens_d"]
            n1, n2 = r["count_real"], r["count_fake"]
            lo, hi = ci_cohens_d_normal_approx(d, n1, n2)
            sub.at[idx, "lo"] = lo
            sub.at[idx, "hi"] = hi

        sub = sub.sort_values("cohens_d")
        y = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(6, max(3, 0.28*len(sub)+1)))
        ax.hlines(y, sub["lo"], sub["hi"], color="gray", alpha=0.7)
        ax.plot(sub["cohens_d"], y, "o")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["dataset"])
        ax.set_xlabel("Cohen's d (Real - Fake)")
        ax.set_title(f"Forest plot by dataset @ Noise STD={ns}")
        save_tight(fig, os.path.join(OUTDIR, f"forest_cohens_d_noise_{ns}.png"))

plot_forest_by_noise_std(pivot)

# 3) 热图：dataset × noise_std（色值 = gap）
def plot_heatmap_gap(pivot_df: pd.DataFrame):
    # 构矩阵
    mat = pivot_df.pivot(index="dataset", columns="noise_std", values="gap_mean_real_minus_fake")
    ds = mat.index.tolist()
    ns_vals = mat.columns.tolist()
    data = mat.values

    fig, ax = plt.subplots(figsize=(1.2*len(ns_vals)+2, 0.35*len(ds)+2))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(np.arange(len(ns_vals)))
    ax.set_xticklabels([str(x) for x in ns_vals], rotation=0)
    ax.set_yticks(np.arange(len(ds)))
    ax.set_yticklabels(ds)
    ax.set_xlabel("Noise STD")
    ax.set_title("Gap (Real - Fake) heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_tight(fig, os.path.join(OUTDIR, "heatmap_gap_dataset_by_noise.png"))

plot_heatmap_gap(pivot)

# 4) 漏斗图：样本量 vs 效应量（gap 或 d）
def plot_funnel(pivot_df: pd.DataFrame, use_metric="gap_mean_real_minus_fake"):
    sub = pivot_df.copy()
    # x：效应量；y：1/sqrt(total_n)
    total_n = sub["count_real"].fillna(0) + sub["count_fake"].fillna(0)
    y = 1 / np.sqrt(total_n.replace(0, np.nan))
    x = sub[use_metric].values

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel(use_metric)
    ax.set_ylabel("1 / sqrt(n_real + n_fake)")
    ax.set_title(f"Funnel plot ({use_metric})")
    save_tight(fig, os.path.join(OUTDIR, f"funnel_{use_metric}.png"))

plot_funnel(pivot, use_metric="gap_mean_real_minus_fake")
plot_funnel(pivot, use_metric="cohens_d")

# 5) 覆盖度：每个 dataset × noise_std 是否有 Real/Fake
def plot_coverage(df_long: pd.DataFrame):
    # 统计 count>0 的覆盖情况：分别对 Real / Fake 作热图
    for t in ["Real", "Fake"]:
        tmp = df_long[df_long["image_type"].str.lower() == t.lower()]
        mat = tmp.pivot(index="dataset", columns="noise_std", values="count")
        ds = mat.index.tolist()
        ns_vals = mat.columns.tolist()
        data = (mat.values > 0).astype(float)  # 1 有覆盖 / 0 无覆盖

        fig, ax = plt.subplots(figsize=(1.1*len(ns_vals)+2, 0.34*len(ds)+2))
        im = ax.imshow(data, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(ns_vals)))
        ax.set_xticklabels([str(x) for x in ns_vals])
        ax.set_yticks(np.arange(len(ds)))
        ax.set_yticklabels(ds)
        ax.set_xlabel("Noise STD")
        ax.set_title(f"Coverage heatmap: {t}")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["No", "Yes"])
        save_tight(fig, os.path.join(OUTDIR, f"coverage_{t.lower()}.png"))

plot_coverage(df)

print("\n=== 完成 ===")
print(f"输出目录：{OUTDIR}")
print("主要文件：")
print("  - GaussianNoiseSimilarity_wide.csv （合并/效应量/显著性等）")
print("  - dataset_level_metrics.csv （数据集层面的斜率、单调性、翻转点、最大差值）")
print("  - gap_curves_by_dataset.png / forest_* / heatmap_* / funnel_* / coverage_*.png")
