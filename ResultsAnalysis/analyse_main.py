"""
一键完成：
1. 搜索所有 epoch*_results.csv
2. 构建矩阵 ap_matrix / acc0.5_matrix / accbest_matrix
3. 分析矩阵：最佳数据集 / 最佳 epoch / per-dataset 最佳 epoch
4. 输出：
   - ap_matrix.csv / acc0.5_matrix.csv / accbest_matrix.csv
   - best_results.csv
   - summary_report.md
   - 图表：热力图、均值曲线、数据集均值条形图

用户只需修改 INPUT_ROOT / OUTPUT_DIR.
"""

import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ==========================================================
#                路径配置（只改这里即可）
# ==========================================================
# 结果根目录（内部应该包含若干 epoch*/epoch*_results.csv）
INPUT_ROOT = r"./TestResults/Progan4cls_train/fdmas/image-denoised-clip"

# 输出路径（矩阵 + 分析报告 + 图表）
OUTPUT_DIR = INPUT_ROOT.replace("TestResults","ResultsAnalysis")

# ==========================================================
#                     配置常量
# ==========================================================
METRIC_MAP = {
    "AP": "ap_matrix.csv",
    "Acc(0.5)": "acc0.5_matrix.csv",
    "Acc(best)": "accbest_matrix.csv",
}

EPOCH_RE = re.compile(r"epoch[_\-]?(\d+)", re.IGNORECASE)


# ==========================================================
#                Part 1: 构建矩阵 (summarize)
# ==========================================================

def extract_epoch_label(path: str) -> str:
    """从路径中提取 epoch 名称"""
    m = EPOCH_RE.search(path.replace("\\", "/"))
    if not m:
        m = EPOCH_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"无法从路径提取 epoch：{path}")
    return f"epoch{int(m.group(1))}"

def find_all_result_files(input_root: str) -> list[str]:
    pattern = os.path.join(input_root, "**", "epoch*", "epoch*_results.csv")
    files = [f for f in glob(pattern, recursive=True) if os.path.isfile(f)]
    if not files:
        pattern2 = os.path.join(input_root, "**", "epoch*", "*.csv")
        files = [f for f in glob(pattern2, recursive=True) if os.path.isfile(f)]
    return sorted(files)

def build_matrices(files: list[str]) -> dict[str, pd.DataFrame]:
    matrices = {}

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] 无法读取：{fpath} | {e}")
            continue

        if "dataset" not in df.columns:
            alt = [c for c in df.columns if c.lower()=="dataset"]
            if not alt:
                continue
            df = df.rename(columns={alt[0]: "dataset"})

        row_label = extract_epoch_label(fpath)

        for metric_col, _outfile in METRIC_MAP.items():
            if metric_col not in df.columns:
                continue

            s = df.set_index("dataset")[metric_col]
            s.name = row_label

            if metric_col not in matrices:
                matrices[metric_col] = pd.DataFrame()

            # 扩展列集合
            matrices[metric_col] = matrices[metric_col].reindex(
                columns=sorted(set(matrices[metric_col].columns).union(s.index))
            )
            matrices[metric_col].loc[row_label, s.index] = s.values

    # 排序与添加 MEAN
    def epoch_key(e):
        m = EPOCH_RE.search(e)
        return int(m.group(1)) if m else 10**9

    for metric, mat in matrices.items():
        # 行按 epoch 数排序
        mat.sort_index(key=lambda idx: [epoch_key(x) for x in idx], inplace=True)
        # 转数值
        for c in mat.columns:
            mat[c] = pd.to_numeric(mat[c], errors="coerce")
        mat["MEAN"] = mat.mean(axis=1, skipna=True)

    return matrices


# ==========================================================
#                Part 2：矩阵分析与可视化
# ==========================================================

def load_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)

    # epoch 排序
    df = df.sort_index(
        key=lambda idx: [int(EPOCH_RE.search(i).group(1)) if EPOCH_RE.search(i) else 10**9 for i in idx]
    )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def best_dataset_by_metric(df: pd.DataFrame):
    cols = [c for c in df.columns if c.upper()!="MEAN"]
    col_means = df[cols].mean(axis=0)
    max_val = col_means.max()
    best_datasets = sorted(col_means[col_means==max_val].index.tolist())
    ranking = col_means.sort_values(ascending=False)
    return ranking, best_datasets, max_val

def best_epoch_by_metric(df: pd.DataFrame):
    row_means = df["MEAN"]
    max_val = row_means.max()
    best_epochs = sorted(row_means[row_means==max_val].index.tolist(),
                         key=lambda x: int(EPOCH_RE.search(x).group(1)))
    return best_epochs, max_val

def best_epoch_per_dataset(df: pd.DataFrame):
    out = {}
    cols = [c for c in df.columns if c.upper()!="MEAN"]
    for col in cols:
        series = df[col]
        vmax = series.max()
        best_eps = [idx for idx, val in series.items() if val==vmax]
        best_eps = sorted(best_eps, key=lambda x: int(EPOCH_RE.search(x).group(1)))
        out[col] = {"best_value": vmax, "best_epochs": best_eps}
    return out


# ---------------- 图表 ---------------- #

def save_heatmap(df: pd.DataFrame, title: str, outpath: str):
    cols = [c for c in df.columns if c.upper()!="MEAN"]
    data = df[cols].to_numpy(float)

    fig, ax = plt.subplots(figsize=(max(8, len(cols)*0.6), max(6, len(df)*0.35)))
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Epoch")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close()

def save_mean_curve(df: pd.DataFrame, title: str, outpath: str):
    epochs = np.array([int(EPOCH_RE.search(x).group(1)) for x in df.index])
    y = df["MEAN"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, y, marker="o")
    ax.set_title(f"{title} - MEAN over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MEAN")

    # 稀疏刻度
    if len(epochs) <= 15:
        xticks = epochs
    else:
        step = max(1, len(epochs)//10)
        xticks = np.unique(np.r_[epochs[0], epochs[::step], epochs[-1]])
    ax.set_xticks(xticks)

    # 标注最高/最低
    imax = np.nanargmax(y)
    imin = np.nanargmin(y)
    ax.scatter([epochs[imax]], [y[imax]])
    ax.scatter([epochs[imin]], [y[imin]])
    ax.annotate(f"max {y[imax]:.2f} @epoch{epochs[imax]}", (epochs[imax], y[imax]),
                textcoords="offset points", xytext=(8, 8))
    ax.annotate(f"min {y[imin]:.2f} @epoch{epochs[imin]}", (epochs[imin], y[imin]),
                textcoords="offset points", xytext=(8, -18))

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close()

def save_dataset_mean_bar(ranking: pd.Series, title: str, outpath: str):
    fig, ax = plt.subplots(figsize=(max(8, len(ranking)*0.5), 4.5))
    ax.bar(ranking.index, ranking.values)
    ax.set_title(f"{title} - Dataset Means")
    ax.set_xticklabels(ranking.index, rotation=45, ha="right")

    # 标数值
    for i, v in enumerate(ranking.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close()


# ==========================================================
#                       主流程
# ==========================================================

def main():
    out_root = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_root, exist_ok=True)

    # ------------------- Part 1: summarize -------------------
    print("==> 搜索 CSV 结果文件...")
    files = find_all_result_files(INPUT_ROOT)
    if not files:
        print("[ERROR] 未找到任何 CSV 文件")
        return
    print(f"[INFO] 找到 {len(files)} 个结果文件，开始构建矩阵…")

    matrices = build_matrices(files)

    # 写出 3 个矩阵
    for metric, fname in METRIC_MAP.items():
        if metric not in matrices:
            print(f"[WARN] 缺少指标：{metric}")
            continue
        path = os.path.join(out_root, fname)
        matrices[metric].to_csv(path, float_format="%.2f", index_label="epoch")
        print(f"[OK] 导出矩阵：{path}")

    # ------------------- Part 2: analysis -------------------
    print("\n==> 开始分析矩阵…")
    analysis_dir = os.path.join(out_root, "PlotAnalysis")
    os.makedirs(analysis_dir, exist_ok=True)

    summary_rows = []
    md_lines = ["# Matrix Analysis Summary", ""]

    for metric, fname in METRIC_MAP.items():
        path = os.path.join(out_root, fname)
        if not os.path.isfile(path):
            continue
        df = load_matrix(path)

        md_lines.append(f"## {metric}")

        ranking, best_ds, best_ds_val = best_dataset_by_metric(df)
        best_epochs, best_epoch_mean = best_epoch_by_metric(df)
        per_ds_best = best_epoch_per_dataset(df)

        summary_rows.append({
            "metric": metric,
            "best_dataset(s)": ";".join(best_ds),
            "best_dataset_mean": round(best_ds_val, 6),
            "best_epoch(s)": ";".join(best_epochs),
            "best_epoch_MEAN": round(best_epoch_mean, 6),
        })

        md_lines.append(f"- **最佳数据集**：{', '.join(best_ds)} (均值 {best_ds_val:.2f})")
        md_lines.append(f"- **最佳 Epoch**：{', '.join(best_epochs)} (MEAN {best_epoch_mean:.2f})")
        md_lines.append("")
        md_lines.append("| Dataset | Best Value | Best Epoch(s) |")
        md_lines.append("|---|---:|---|")
        for ds, info in per_ds_best.items():
            md_lines.append(f"| {ds} | {info['best_value']:.2f} | {', '.join(info['best_epochs'])} |")
        md_lines.append("")

        # 图表
        save_heatmap(df, f"{metric} Heatmap",
                     os.path.join(analysis_dir, f"{metric}_heatmap.png"))
        save_mean_curve(df, metric,
                        os.path.join(analysis_dir, f"{metric}_mean_curve.png"))
        save_dataset_mean_bar(ranking, metric,
                        os.path.join(analysis_dir, f"{metric}_dataset_mean_bar.png"))

    # 写出 best_results.csv
    best_csv = os.path.join(analysis_dir, "best_results.csv")
    pd.DataFrame(summary_rows).to_csv(best_csv, index=False)
    print(f"[OK] 输出最佳汇总：{best_csv}")

    # 写出 Markdown 报告
    md_path = os.path.join(analysis_dir, "summary_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] 输出报告：{md_path}")

    print("\n[DONE] 所有处理已完成！图表 / 矩阵 / 汇总已生成。")


if __name__ == "__main__":
    main()
