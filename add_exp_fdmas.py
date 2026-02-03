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

MultiDatasetName = ['fdmas']

class RealFakeDataset(Dataset):
    """
    自适应多种目录结构的数据集读取器。
    兼容：
      A) root/{0_real|1_fake}/{image}
      B) root/{class}/{0_real|1_fake}/{image}
      C) 更深一层也可，凡是目录名刚好为 0_real 或 1_fake 都会被识别
    """
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 0 for real, 1 for fake

        print(f"正在从 {root_dir} 加载数据集...")

        # 在任意层级查找名为 0_real 或 1_fake 的目录
        real_dirs, fake_dirs = [], []
        for dirpath, dirnames, _ in os.walk(root_dir):
            # 标准化比较，避免不同大小写或者尾部分隔符影响
            names = {d.strip() for d in dirnames}
            if '0_real' in names:
                real_dirs.append(os.path.join(dirpath, '0_real'))
            if '1_fake' in names:
                fake_dirs.append(os.path.join(dirpath, '1_fake'))

        # 若根目录本身就是 0_real/1_fake 之一，也要识别
        base = os.path.basename(os.path.normpath(root_dir))
        if base == '0_real':
            real_dirs.append(root_dir)
        elif base == '1_fake':
            fake_dirs.append(root_dir)

        # 收集图像
        def collect_from(d, label):
            try:
                for name in os.listdir(d):
                    if name.lower().endswith(self.IMG_EXTS):
                        self.image_paths.append(os.path.join(d, name))
                        self.labels.append(label)
            except Exception as e:
                print(f"警告：无法读取目录 {d}: {e}")

        for d in real_dirs:
            collect_from(d, 0)
        for d in fake_dirs:
            collect_from(d, 1)

        print(f"数据集加载完成，共找到 {len(self.image_paths)} 张图像。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"错误：无法加载图像 {img_path}: {e}")
            return None, -1


def apply_jpeg_compression(pil_image, quality=75):
    """对 PIL 图像应用 JPEG 压缩"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    """
    对 PIL.Image 添加零均值高斯噪声，std 为 [0,1] 区间的相对标准差。
    - 灰度图: 直接加噪
    - RGB: 对 3 个通道加噪
    - RGBA: 仅对 RGB 加噪，alpha 通道保持不变
    """
    arr = np.asarray(img).astype(np.float32) / 255.0  # -> [0,1]
    if arr.ndim == 2:  # H,W  -> H,W,1
        arr = arr[..., None]

    H, W, C = arr.shape
    if C == 4:
        rgb = arr[..., :3]
        a   = arr[..., 3:]
        noise = np.random.normal(loc=0.0, scale=std, size=rgb.shape).astype(np.float32)
        rgb_noised = np.clip(rgb + noise, 0.0, 1.0)
        out = np.concatenate([rgb_noised, a], axis=-1)
    else:
        noise = np.random.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0.0, 1.0)

    out = (out * 255.0).round().astype(np.uint8)
    if out.shape[2] == 1:
        out = out[..., 0]  # 回到 H,W
    return Image.fromarray(out)


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


def looks_like_single_dataset(root: str) -> bool:
    """
    判断 root 是否为单数据集根：
    结构应为 root/{class}/(0_real|1_fake)/image.png
    """
    if not os.path.isdir(root):
        return False
    try:
        for cls in os.listdir(root):
            cls_path = os.path.join(root, cls)
            if os.path.isdir(cls_path):
                has_real = os.path.isdir(os.path.join(cls_path, "0_real"))
                has_fake = os.path.isdir(os.path.join(cls_path, "1_fake"))
                # 只要发现一个 class 目录下存在 0_real 或 1_fake，即可判定为单数据集结构
                if has_real or has_fake:
                    return True
        return False
    except Exception:
        return False


def iter_fdmas_subdatasets(fdmas_root: str):
    """
    在 fdmas 根目录下发现所有有效子数据集目录（满足 looks_like_single_dataset）。
    返回 (ds_name, ds_path) 生成器。
    """
    if not os.path.isdir(fdmas_root):
        return
    for name in sorted(os.listdir(fdmas_root)):
        ds_path = os.path.join(fdmas_root, name)
        if os.path.isdir(ds_path) and looks_like_single_dataset(ds_path):
            yield name, ds_path


def process_one_dataset(ds_dir, ds_name, args, model, device, clip_preprocess, resize_transform):
    print(f"\n===== 正在处理数据集：{ds_name} =====")
    sub_out = os.path.join(args.output_dir, ds_name)
    os.makedirs(sub_out, exist_ok=True)

    dataset = RealFakeDataset(root_dir=ds_dir, transform=None)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn)

    if args.transform_type == 'jpeg':
        params_to_test, param_name = args.jpeg_quality, "JPEG Quality"
    elif args.transform_type == 'blur':
        params_to_test, param_name = args.blur_sigma, "Blur Sigma"
    elif args.transform_type == 'noise':
        params_to_test, param_name = args.noise_std, "Noise STD"
    else:
        raise ValueError(f"Unknown transform_type: {args.transform_type}")


    print(f"\n开始对 '{args.transform_type}' 变换进行多参数分析（数据集：{ds_name}）...")
    for param_val in tqdm(params_to_test, desc=f"{ds_name} | 总进度"):
        tqdm.write(f"--- [{ds_name}] 测试参数: {param_name} = {param_val} ---")

        if args.transform_type == 'jpeg':
            post_transform = lambda img, q=param_val: apply_jpeg_compression(img, quality=q)
        elif args.transform_type == 'blur':
            post_transform = transforms.GaussianBlur(kernel_size=args.blur_kernel, sigma=param_val)
        elif args.transform_type == 'noise':
            post_transform = lambda img, s=param_val: apply_gaussian_noise(img, std=s)
        else:
            raise ValueError(f"Unknown transform_type: {args.transform_type}")


        real_sims, fake_sims = run_analysis(
            dataloader=dataloader, model=model, device=device,
            post_transform=post_transform,
            resize_transform=resize_transform,
            clip_preprocess=clip_preprocess
        )

        rows = []
        if len(real_sims) > 0:
            rows.append({'dataset': ds_name, 'parameter': param_name, 'value': param_val,
                         'image_type': 'Real', 'count': len(real_sims),
                         'mean_similarity': np.mean(real_sims), 'std_similarity': np.std(real_sims)})
        if len(fake_sims) > 0:
            rows.append({'dataset': ds_name, 'parameter': param_name, 'value': param_val,
                         'image_type': 'Fake', 'count': len(fake_sims),
                         'mean_similarity': np.mean(fake_sims), 'std_similarity': np.std(fake_sims)})
        df = pd.DataFrame(rows)

        base_plot, ext = os.path.splitext(args.output_plot)
        plot_path = os.path.join(sub_out, f"{base_plot}_{ds_name}_{param_val}{ext}")
        base_csv, cext = os.path.splitext(args.output_csv)
        csv_path = os.path.join(sub_out, f"{base_csv}_{ds_name}_{param_val}{cext}")

        if not df.empty:
            tqdm.write(df.to_string(index=False, float_format="%.4f"))
        df.to_csv(csv_path, index=False)
        tqdm.write(f"统计数据已保存至: {csv_path}")

        plt.figure(figsize=(12, 7))
        plt.hist(real_sims, bins=100, alpha=0.7, label='Real Images', density=True, range=(0.85, 1.0))
        plt.hist(fake_sims, bins=100, alpha=0.7, label='Fake Images', density=True, range=(0.85, 1.0))
        plt.title(f'Cosine Similarity Distribution (Dataset: {ds_name}, '
                  f'Transform: {args.transform_type.capitalize()}, {param_name}={param_val})', fontsize=16)
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        plt.savefig(plot_path)
        plt.close()
        tqdm.write(f"分析图表已保存至: {plot_path}\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"正在加载 CLIP 模型: {args.clip_model}")
    model = FeatureMetric(f"CLIP:{args.clip_model}").to(device)
    model.eval()

    clip_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    resize_transform = transforms.Resize((224, 224), interpolation=Image.BICUBIC)

    os.makedirs(args.output_dir, exist_ok=True)

    root = args.data_dir
    print(f"数据根目录: {root}")



    if args.data_name in MultiDatasetName:
        # 合集模式：遍历下一层所有子目录，逐个按“原先代码”流程评测
        subdirs = [d for d in sorted(os.listdir(root))
                   if os.path.isdir(os.path.join(root, d))]
        if not subdirs:
            print("未在 fdmas 根下发现任何子数据集目录。")
        for ds_name in subdirs:
            ds_dir = os.path.join(root, ds_name)
            process_one_dataset(
                ds_dir=ds_dir, ds_name=ds_name, args=args,
                model=model, device=device,
                clip_preprocess=clip_preprocess,
                resize_transform=resize_transform
            )
    else:
        # 单数据集模式：例如 progan / cyclegan / ADM / DALLE2 ...
        ds_name = os.path.basename(os.path.normpath(root))
        process_one_dataset(
            ds_dir=root, ds_name=ds_name, args=args,
            model=model, device=device,
            clip_preprocess=clip_preprocess,
            resize_transform=resize_transform
        )

    print("所有参数分析完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析 CLIP 特征在变换下的余弦相似度。')
    parser.add_argument('--data_name', type=str, required=True, help='数据集名称,fdmas,progan。')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集的根目录。')

    # --- 修改输出参数 ---
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='存放所有输出文件的目录。')
    parser.add_argument('--output_plot', type=str, default='similarity_distribution.pdf', help='输出图表的基础文件名。')
    parser.add_argument('--output_csv', type=str, default='similarity_stats.csv',
                        help='输出统计数据 CSV 文件的基础文件名。')

    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='要使用的 CLIP 模型名称。')
    parser.add_argument('--batch_size', type=int, default=32, help='处理数据的批次大小。')
    parser.add_argument('--num_workers', type=int, default=16, help='Dataloader 使用的进程数。')
    # 高斯噪声
    parser.add_argument('--transform_type', type=str,
                        choices=['jpeg', 'blur', 'noise'], required=True,
                        help='要应用的变换类型 (jpeg / blur / noise)。')
    # --- 高斯噪声的参数（相对强度标准差，定义在[0,1]） ---
    parser.add_argument('--noise_std', type=float, nargs='+', default=[0.03, 0.05, 0.10],
                        help='一个或多个高斯噪声标准差(相对[0,1]强度)，如 0.03 0.05 0.10。')


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
# python add_exp_fdmas.py --data_name fdmas --data_dir ../Datasets/fdmas/test --transform_type blur --blur_sigma 1 --output_dir add_exp/fdmas_1030/GaussianBlur --output_csv blur_coscine_similarity.csv

# progan完整 高斯噪声
# python add_exp_fdmas.py --data_name progan --data_dir ../Datasets/ --transform_type noise --noise_std 0.03 0.05 0.1 --output_dir add_exp/progan/GaussianNoise --output_csv gaussiannoise_coscine_similarity.csv

# Chameleon/test 高斯噪声
# python add_exp_fdmas.py --data_name Chameleon --data_dir ../Datasets/Chameleon/test --transform_type noise --noise_std 0.03 0.05 0.1 --output_dir add_exp/Chameleon/GaussianNoise --output_csv gaussiannoise_coscine_similarity.csv
