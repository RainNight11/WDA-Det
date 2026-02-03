import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import io
import pandas as pd
from itertools import cycle

# 假设 models_v2.py 在 'models' 目录下
# 这个类用于加载一个冻结的 CLIP 模型来提取特征
from models.models_v2 import FeatureMetric


class RealFakeDataset(Dataset):
    """
    一个自定义的数据集类，用于加载具有特定目录结构的数据。
    结构: data_dir/{class_name}/{0_real/1_fake}/{image.png}
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 0 for real, 1 for fake

        print(f"正在从 {root_dir} 加载数据集...")
        # 遍历目录以查找图像
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # 遍历 '0_real' 和 '1_fake' 子目录
                for real_fake_dir in ['0_real', '1_fake']:
                    sub_dir = os.path.join(class_path, real_fake_dir)
                    if os.path.isdir(sub_dir):
                        label = 0 if real_fake_dir == '0_real' else 1
                        for img_name in os.listdir(sub_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.image_paths.append(os.path.join(sub_dir, img_name))
                                self.labels.append(label)
        print(f"数据集加载完成，共找到 {len(self.image_paths)} 张图像。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # 以 PIL 格式打开图像
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"错误：无法加载图像 {img_path}: {e}")
            # 返回一个虚拟数据以避免 Dataloader 崩溃
            return None, -1


def apply_jpeg_compression(pil_image, quality=75):
    """对 PIL 图像应用 JPEG 压缩"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def collate_fn(batch):
    """
    自定义的 collate_fn，用于过滤掉加载失败的图像，并返回 PIL 图像列表。
    """
    # 过滤掉加载失败的项目 (图像为 None)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        # 如果整个批次都无效，则返回空列表
        return [], []

    # 将图像和标签解压到各自的列表中
    images, labels = zip(*batch)
    return list(images), list(labels)  # 返回 PIL 图像列表和标签列表


def run_analysis(dataloader, model, device, post_transform, resize_transform, clip_preprocess):
    """为给定的变换运行分析的核心逻辑"""
    real_sims = []
    fake_sims = []

    for pil_images, labels_list in tqdm(dataloader, desc="正在处理批次", leave=False):
        if not pil_images:
            continue

        original_tensors = []
        transformed_tensors = []

        for pil_image in pil_images:
            resized_pil = resize_transform(pil_image)
            transformed_pil = post_transform(resized_pil)
            original_tensors.append(clip_preprocess(resized_pil))
            transformed_tensors.append(clip_preprocess(transformed_pil))

        original_batch = torch.stack(original_tensors).to(device)
        transformed_batch = torch.stack(transformed_tensors).to(device)
        labels = torch.tensor(labels_list, device=device)

        with torch.no_grad():
            feat_orig = model(original_batch)
            feat_transformed = model(transformed_batch)
            similarities = F.cosine_similarity(feat_orig, feat_transformed, dim=1)

        for i in range(similarities.shape[0]):
            if labels[i] == 0:
                real_sims.append(similarities[i].item())
            elif labels[i] == 1:
                fake_sims.append(similarities[i].item())

    return np.array(real_sims), np.array(fake_sims)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 CLIP 模型
    print(f"正在加载 CLIP 模型: {args.clip_model}")
    model = FeatureMetric(f"CLIP:{args.clip_model}").to(device)
    model.eval()

    # CLIP 模型的官方预处理流程
    clip_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    resize_transform = transforms.Resize((224, 224), interpolation=Image.BICUBIC)

    # 加载数据集 (不应用任何变换)
    dataset = RealFakeDataset(root_dir=args.data_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    # --- 创建输出目录 ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 循环处理参数 ---
    params_to_test = []
    if args.transform_type == 'jpeg':
        params_to_test = args.jpeg_quality
        param_name = "JPEG Quality"
    else:  # 'blur'
        params_to_test = args.blur_sigma
        param_name = "Blur Sigma"

    print(f"\n开始对 '{args.transform_type}' 变换进行多参数分析...")
    for param_val in tqdm(params_to_test, desc="总进度"):
        tqdm.write(f"--- 正在测试参数: {param_name} = {param_val} ---")

        # 定义变换
        if args.transform_type == 'jpeg':
            post_transform = lambda img: apply_jpeg_compression(img, quality=param_val)
        else:  # 'blur'
            post_transform = transforms.GaussianBlur(kernel_size=args.blur_kernel, sigma=param_val)

        # 运行分析
        real_sims, fake_sims = run_analysis(dataloader, model, device, post_transform, resize_transform,
                                            clip_preprocess)

        # --- 为当前参数生成独立的报告 ---

        # 准备统计数据
        current_results = []
        if len(real_sims) > 0:
            current_results.append({
                'parameter': param_name, 'value': param_val, 'image_type': 'Real',
                'count': len(real_sims), 'mean_similarity': np.mean(real_sims), 'std_similarity': np.std(real_sims)
            })
        if len(fake_sims) > 0:
            current_results.append({
                'parameter': param_name, 'value': param_val, 'image_type': 'Fake',
                'count': len(fake_sims), 'mean_similarity': np.mean(fake_sims), 'std_similarity': np.std(fake_sims)
            })

        df = pd.DataFrame(current_results)

        # 打印当前参数的表格
        tqdm.write(f"\n--- 参数 {param_name} = {param_val} 的统计结果 ---")
        tqdm.write(df.to_string(index=False, float_format="%.4f"))

        # 生成独立的文件名
        base_plot_name, plot_ext = os.path.splitext(args.output_plot)
        plot_filename = f"{base_plot_name}_{param_val}{plot_ext}"
        output_plot_path = os.path.join(args.output_dir, plot_filename)

        base_csv_name, csv_ext = os.path.splitext(args.output_csv)
        csv_filename = f"{base_csv_name}_{param_val}{csv_ext}"
        output_csv_path = os.path.join(args.output_dir, csv_filename)

        # 保存到独立的 CSV
        df.to_csv(output_csv_path, index=False)
        tqdm.write(f"统计数据已保存至: {output_csv_path}")

        # --- 为当前参数绘制独立的图表 ---
        plt.figure(figsize=(12, 7))
        plt.hist(real_sims, bins=100, alpha=0.7, label='Real Images', density=True, range=(0.85, 1.0))
        plt.hist(fake_sims, bins=100, alpha=0.7, label='Fake Images', density=True, range=(0.85, 1.0))
        plt.title(
            f'Cosine Similarity Distribution (Transform: {args.transform_type.capitalize()}, {param_name}={param_val})',
            fontsize=16)
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        plt.savefig(output_plot_path)
        plt.close()  # 关闭当前图表，避免在下一次循环中重叠
        tqdm.write(f"分析图表已保存至: {output_plot_path}\n")

    print("所有参数分析完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析 CLIP 特征在变换下的余弦相似度。')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集的根目录。')
    parser.add_argument('--transform_type', type=str, choices=['jpeg', 'blur'], required=True,
                        help='要应用的变换类型 (jpeg 或 blur)。')

    # --- 修改输出参数 ---
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='存放所有输出文件的目录。')
    parser.add_argument('--output_plot', type=str, default='similarity_distribution.pdf', help='输出图表的基础文件名。')
    parser.add_argument('--output_csv', type=str, default='similarity_stats.csv',
                        help='输出统计数据 CSV 文件的基础文件名。')

    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='要使用的 CLIP 模型名称。')
    parser.add_argument('--batch_size', type=int, default=32, help='处理数据的批次大小。')
    parser.add_argument('--num_workers', type=int, default=16, help='Dataloader 使用的进程数。')

    # --- 修改为接受列表 ---
    parser.add_argument('--jpeg_quality', type=int, nargs='+', default=[75],
                        help='一个或多个 JPEG 压缩质量 (1-100)，以空格分隔。')
    parser.add_argument('--blur_sigma', type=float, nargs='+', default=[5.0],
                        help='一个或多个高斯模糊的 sigma 值，以空格分隔。')

    # --- 高斯模糊的固定参数 ---
    parser.add_argument('--blur_kernel', type=int, default=21, help='高斯模糊的核大小 (保持固定)。')


    args = parser.parse_args()
    main(args)

# python add_exp.py --data_dir ../Datasets/progan_1percent/train/progan --transform_type jpeg --jpeg_quality 95 75 50 --output_dir add_exp
# python add_exp.py --data_dir ../Datasets/progan_1percent/train/progan --transform_type blur --blur_sigma 0.7 1 1.2 --output_dir add_exp
# python add_exp.py --data_dir ../Datasets/progan_1percent/train/progan --transform_type blur --blur_sigma 1 2 3 5 --output_dir add_exp
# python add_exp.py --data_dir ../Datasets/fdmas_sample --transform_type blur --blur_sigma 1 2 3 5 --output_dir add_exp/fdmas_sample/GaussianBlur --output_csv blur_coscine_similarity.csv
# python add_exp.py --data_dir ../Datasets/fdmas_sample --transform_type jpeg --jpeg_quality 95 75 50 --output_dir add_exp/fdmas_sample/JPEG --output_csv jpeg_coscine_similarity.csv
# python add_exp.py --data_dir ../Datasets/fdmas/test --transform_type blur --blur_sigma 1 --output_dir add_exp/fdmas/GaussianBlur --output_csv blur_coscine_similarity.csv