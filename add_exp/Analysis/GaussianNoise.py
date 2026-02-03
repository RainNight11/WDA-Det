import os
import re
import glob
import math
import pandas as pd

ROOT = "../fdmas_1101/GaussianNoise"  # 当前目录为 GaussianNoise/
OUT_LONG = "GaussianNoiseSimilarity.csv"
OUT_WIDE = "GaussianNoiseSimilarity_wide.csv"

# 匹配文件名中的 dataset 与噪声 std（例如: noise_cosine_similarity_ADM_0.05.csv）
FILE_RE = re.compile(r"noise_cosine_similarity_(?P<dataset>.+)_(?P<std>0\.\d+|1(?:\.0)?)\.csv$", re.IGNORECASE)

def read_one_csv(path):
    df = pd.read_csv(path)
    # 基本列校验
    required = ["dataset","parameter","value","image_type","count","mean_similarity","std_similarity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")

    # 从文件名提取 dataset/std 并与内容交叉校验（容错：不一致时以内容为准但记录提示）
    m = FILE_RE.search(os.path.basename(path))
    if m:
        fn_dataset = m.group("dataset")
        fn_std = m.group("std")
        unique_dataset = df["dataset"].astype(str).str.strip().unique()
        unique_std = df["value"].astype(str).str.strip().unique()
        warn = []
        if len(unique_dataset)==1 and unique_dataset[0] != fn_dataset:
            warn.append(f"文件名数据集={fn_dataset}, 表内数据集={unique_dataset[0]}")
        if len(unique_std)==1 and unique_std[0] != fn_std:
            warn.append(f"文件名std={fn_std}, 表内value={unique_std[0]}")
        if warn:
            print(f"[WARN] {path}: " + " | ".join(warn))
    else:
        print(f"[WARN] 无法从文件名解析 dataset/std: {path}")

    # 统一类型
    df["dataset"] = df["dataset"].astype(str)
    df["parameter"] = df["parameter"].astype(str)
    df["image_type"] = df["image_type"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["mean_similarity"] = pd.to_numeric(df["mean_similarity"], errors="coerce")
    df["std_similarity"] = pd.to_numeric(df["std_similarity"], errors="coerce")

    # 增加源文件列，便于追溯
    df["source_csv"] = os.path.relpath(path, ROOT)
    # 把 value 命名清晰些
    df = df.rename(columns={"value":"noise_std"})
    return df

def cohen_d(m1, s1, n1, m2, s2, n2):
    # Hedges' g 更严格，但这里先用 Cohen's d（合并标准差）
    if any(x in (None, float("nan")) for x in [s1,s2,n1,n2]) or min(n1,n2) <= 1:
        return float("nan")
    # 合并方差（无偏估计可再做修正）
    sp2 = ((n1-1)*(s1**2) + (n2-1)*(s2**2)) / (n1+n2-2) if (n1+n2-2) > 0 else float("nan")
    if sp2 <= 0 or math.isnan(sp2):
        return float("nan")
    sp = math.sqrt(sp2)
    if sp == 0:
        return float("nan")
    return (m1 - m2) / sp

def main():
    # 扫描所有 CSV
    csvs = glob.glob(os.path.join(ROOT, "**", "noise_cosine_similarity_*.csv"), recursive=True)
    if not csvs:
        raise SystemExit("未找到任何 noise_cosine_similarity_*.csv 文件")

    frames = []
    for p in sorted(csvs):
        try:
            frames.append(read_one_csv(p))
        except Exception as e:
            print(f"[ERROR] 读取失败: {p} -> {e}")

    long_df = pd.concat(frames, ignore_index=True)
    # 仅保留参数为噪声的记录（保险起见）
    long_df = long_df[long_df["parameter"].str.lower().str.contains("noise")]

    # 排序、输出长表
    long_df = long_df.sort_values(["dataset","noise_std","image_type"]).reset_index(drop=True)
    long_df.to_csv(OUT_LONG, index=False)
    print(f"[OK] 已写出长表: {OUT_LONG}  （{len(long_df)} 行）")

    # 生成 wide 对比表：每个 (dataset, noise_std) 形成 Real/Fake 两列 + 差值/效应量
    piv = long_df.pivot_table(
        index=["dataset","noise_std"],
        columns="image_type",
        values=["mean_similarity","std_similarity","count"],
        aggfunc="first"
    )

    # 扁平化列名
    piv.columns = [f"{a}_{b}".lower() for a,b in piv.columns]
    piv = piv.reset_index()

    # 计算差值与 Cohen's d（Real - Fake）
    for col in ["mean_similarity", "std_similarity", "count"]:
        for t in ["real","fake"]:
            if f"{col}_{t}" not in piv.columns:
                piv[f"{col}_{t}"] = float("nan")

    piv["gap_mean_real_minus_fake"] = piv["mean_similarity_real"] - piv["mean_similarity_fake"]
    piv["cohen_d"] = piv.apply(
        lambda r: cohen_d(
            r["mean_similarity_real"], r["std_similarity_real"], r["count_real"],
            r["mean_similarity_fake"], r["std_similarity_fake"], r["count_fake"]
        ),
        axis=1
    )

    # 排序并输出
    piv = piv.sort_values(["dataset","noise_std"]).reset_index(drop=True)
    piv.to_csv(OUT_WIDE, index=False)
    print(f"[OK] 已写出宽表: {OUT_WIDE}  （{len(piv)} 行）")

if __name__ == "__main__":
    main()
