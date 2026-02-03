# 这个脚本基于add_exp_fdmas.py构建，以下相同均指该脚本内同名函数相同
# 这个脚本我目标是实现输入图像-->加入高斯噪声-->送进模型（这里我可能用的clip，怎么改下面应该是留接口了）-->计算与原图相似度，默认进来图像不知道是real还是fake，如果有问题的话可能就是我听差了
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

from models.pretrained_models import FeatureMetric

# 这个脚本基于add_exp_fdmas.py构建，以下相同均指该脚本内同名函数相同

#拉取函数准备工作外置
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
MultiDatasetName = ['fdmas']

#导入未知图像，此处完全重写，具体内容可在gpt脚本模式问答下看见全部代码，已经手筛过一轮了，不是无脑ctrl+C

class UnlabeledImageDataset(Dataset):
    """
    递归搜集 root_dir 下的所有图片；如果只处理单图，用 image_path。
    返回 (PIL.Image, path)
    """
    def __init__(self, root_dir=None, image_path=None, transform=None):
        self.transform = transform
        self.items = []     #此处修改

        if image_path is not None:
            assert os.path.isfile(image_path), f"找不到图像：{image_path}"
            self.items = [image_path]
            print(f"[unlabeled] 单图模式，共 1 张图：{image_path}")
        else:
            assert root_dir is not None and os.path.isdir(root_dir), f"无效目录：{root_dir}"
            print(f"[unlabeled] 正在递归扫描目录：{root_dir}")
            for dirpath, _, filenames in os.walk(root_dir):
                for name in filenames:
                    if name.lower().endswith(IMG_EXTS):
                        self.items.append(os.path.join(dirpath, name))
            self.items.sort()
            print(f"[unlabeled] 共找到 {len(self.items)} 张图像。")
#这个函数修改参数
    def __len__(self):
        return len(self.items)
#这个函数修改了，新增p,是个路径字符串
    def __getitem__(self, idx):
        p = self.items[idx]     #在这里
        try:
            image = Image.open(p).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, p
        except Exception as e:
            print(f"错误：无法加载图像 {p}: {e}")
            return None, p
# 下面这两个函数我没动
#虽然不知道为啥要压缩，但还是拷贝过来
def apply_jpeg_compression(pil_image, quality=75):
    """对 PIL 图像应用 JPEG 压缩"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

#高斯噪声，传参有些复杂，具体内容没动，如果有问题考虑参数修改
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

    H, W, C = arr.shape  #AI放弃这段，我仍然保留
    if C == 4:   #此处判断条件可能不同，我没做修改
        rgb = arr[..., :3]
        a   = arr[..., 3:]
        noise = np.random.normal(loc=0.0, scale=std, size=rgb.shape).astype(np.float32)    #该处传参方式AI进行了修改，我没动
        rgb_noised = np.clip(rgb + noise, 0.0, 1.0)
        out = np.concatenate([rgb_noised, a], axis=-1)
    else:
        noise = np.random.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0.0, 1.0)

    out = (out * 255.0).round().astype(np.uint8)
    if out.ndim == 3 and out.shape[2] == 1: #新增判断条件
        out = out[..., 0]  # 回到 H,W
    return Image.fromarray(out)

#打包函数（新）
def collate_unlabeled(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return [], []
    images, paths = zip(*batch)
    return list(images), list(paths)

    # 将图像和标签解压到各自的列表中
    images, labels = zip(*batch)
    return list(images), list(labels)  # 返回 PIL 图像列表和标签列表

#新的余弦函数相似度计算
def cosine_similarities(model, device, clip_preprocess, resize_transform, pil_list_a, pil_list_b):
    original_tensors, transformed_tensors = [], []
    for pa, pb in zip(pil_list_a, pil_list_b):
        ra = resize_transform(pa)
        rb = resize_transform(pb)
        original_tensors.append(clip_preprocess(ra))
        transformed_tensors.append(clip_preprocess(rb))
    with torch.no_grad():
        a = torch.stack(original_tensors).to(device)
        b = torch.stack(transformed_tensors).to(device)
        fa = model(a)
        fb = model(b)
        sims = F.cosine_similarity(fa, fb, dim=1)
    return sims.detach().cpu().numpy()

#功能写下面了，这个我没看懂
#在未知真假标签的情况下，测量每张图片在重复扰动（高斯噪声/JPEG压缩等）后，其特征表示的稳定性，完全新的东西
def run_unlabeled(dataloader, model, device, clip_preprocess, resize_transform,
                  transform_maker, repeats=1):
    """
    transform_maker: 接收参数值 -> 返回“对单张 PIL 做变换”的函数（callable）
    repeats: 对同一图像做多少次随机扰动（>1 时返回均值/标准差更稳定；JPEG/Blur 可设为1）
    返回：一个列表，每个元素是 {path, sims:[...]} 或 {path, mean, std, n}
    """
    results = []
    for pil_images, paths in tqdm(dataloader, desc="(unlabeled) 批次", leave=False):
        if not pil_images:
            continue
        # 对每次 repeat，构造变换后的图像列表
        # 对噪声，多次采样；对 JPEG/Blur，重复采样意义不大（可=1）
        all_sims = []
        for _ in range(repeats):
            transformed = [transform_maker(img) for img in pil_images]
            sims = cosine_similarities(
                model, device, clip_preprocess, resize_transform,
                pil_images, transformed
            )
            all_sims.append(sims)
        all_sims = np.stack(all_sims, axis=0)  # [repeats, B]
        means = all_sims.mean(axis=0)
        stds  = all_sims.std(axis=0, ddof=0)
        for p, m, s in zip(paths, means, stds):
            results.append({"image_path": p, "mean_similarity": float(m),
                            "std_similarity": float(s), "repeats": repeats})
    return results

#核心处理算法，此处处理了不少，但是比较混乱，我反正没看懂逻辑，祈祷是对的
def process_unlabeled(args, model, device, clip_preprocess, resize_transform):
    # 建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    sub_out = args.output_dir  # 无标签就直接放在 output_dir

    # 构造数据集
    if args.image_path is not None:
        dataset = UnlabeledImageDataset(image_path=args.image_path, transform=None)
    else:
        dataset = UnlabeledImageDataset(root_dir=args.data_dir, transform=None)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_unlabeled)

    # 参数列表
    if args.transform_type == 'jpeg':
        params_to_test, param_name = args.jpeg_quality, "JPEG Quality"
    elif args.transform_type == 'blur':
        params_to_test, param_name = args.blur_sigma, "Blur Sigma"
    elif args.transform_type == 'noise':
        params_to_test, param_name = args.noise_std, "Noise STD"
    else:
        raise ValueError(f"Unknown transform_type: {args.transform_type}")

    # 针对每个参数值跑一遍，输出一份 CSV；如果是单图，也会按参数值输出多行
    print(f"\n开始 '{args.transform_type}' 变换多参数分析（unlabeled）...")
    all_summary_rows = []  # 可选：把各参数的全局统计也汇总一下

    for param_val in tqdm(params_to_test, desc="unlabeled | 总进度"):
        tqdm.write(f"--- [unlabeled] 测试参数: {param_name} = {param_val} ---")

        if args.transform_type == 'jpeg':
            transform_maker = lambda img, q=param_val: apply_jpeg_compression(img, quality=q)
            repeats = max(1, args.repeats)  # JPEG 重复意义小，但仍允许 >1
        elif args.transform_type == 'blur':
            blur_op = transforms.GaussianBlur(kernel_size=args.blur_kernel, sigma=param_val)
            transform_maker = lambda img, op=blur_op: op(img)
            repeats = max(1, args.repeats)
        elif args.transform_type == 'noise':
            transform_maker = lambda img, s=param_val: apply_gaussian_noise(img, std=s)
            repeats = max(1, args.repeats)

        per_image = run_unlabeled(
            dataloader, model, device, clip_preprocess, resize_transform,
            transform_maker, repeats=repeats
        )

        # 写 per-image CSV
        df = pd.DataFrame(per_image)
        df.insert(1, 'parameter', param_name)
        df.insert(2, 'value', param_val)

        base_csv, cext = os.path.splitext(args.output_csv_unlabeled)
        csv_path = os.path.join(sub_out, f"{base_csv}_{args.transform_type}_{param_val}{cext}")
        df.to_csv(csv_path, index=False)
        tqdm.write(f"[unlabeled] 逐图结果已保存：{csv_path}")

        # 给个分布图（整体）
        plot_path = os.path.join(sub_out, f"unlabeled_similarity_{args.transform_type}_{param_val}.pdf")
        plt.figure(figsize=(10, 6))
        plt.hist(df["mean_similarity"], bins=100, density=True)
        plt.title(f'Unlabeled Similarity (Transform={args.transform_type}, {param_name}={param_val})')
        plt.xlabel('Cosine Similarity (mean over repeats)')
        plt.ylabel('Density')
        plt.grid(axis='y', linestyle='--')
        plt.savefig(plot_path)
        plt.close()
        tqdm.write(f"[unlabeled] 分布图已保存：{plot_path}\n")

        # 汇总一行全局统计
        all_summary_rows.append({
            "parameter": param_name,
            "value": param_val,
            "count": int(df.shape[0]),
            "mean_of_means": float(df["mean_similarity"].mean()),
            "std_of_means": float(df["mean_similarity"].std(ddof=0)),
            "avg_std_within": float(df["std_similarity"].mean()),
            "repeats": repeats
        })

    # 写全局汇总
    if all_summary_rows:
        sum_df = pd.DataFrame(all_summary_rows)
        sum_path = os.path.join(sub_out, f"unlabeled_summary_{args.transform_type}.csv")
        sum_df.to_csv(sum_path, index=False)
        tqdm.write(f"[unlabeled] 全局汇总已保存：{sum_path}")

#主函数，新增内容在下面，主干也动了不少
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"正在加载 backbone 模型: {args.backbone_model}")
    model = FeatureMetric(args.backbone_model).to(device)
    model.eval()

    clip_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    resize_transform = transforms.Resize((224, 224), interpolation=Image.BICUBIC)

    os.makedirs(args.output_dir, exist_ok=True)

    # ==== 无标签模式（优先判断）====
    if args.unlabeled or args.image_path is not None:
        # 注意：unlabeled 可以搭配 data_dir；或者直接传 image_path 单图
        if not args.image_path and not args.data_dir:
            raise ValueError("无标签模式需要 --data_dir 或 --image_path 之一。")
        process_unlabeled(args, model, device, clip_preprocess, resize_transform)
        print("（unlabeled）所有分析完成。")
        return
#代码入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='分析 CLIP 特征在变换下的余弦相似度（支持 labeled / unlabeled 两种模式）。'
    )

    # —— 通用输入 ——
    parser.add_argument('--data_name', type=str, default='fdmas',
                        help='数据集名称（用于 labeled 模式的集合名，例如 fdmas, progan…）。')
    parser.add_argument('--data_dir', type=str, default='',
                        help='数据根目录（labeled 和 unlabeled 都可用）。')
    parser.add_argument('--image_path', type=str, default=None,
                        help='无标签单图模式：直接传入单张图像路径。')
    parser.add_argument('--unlabeled', action='store_true',
                        help='启用无标签模式（与 --image_path 二选一，或都给则优先单图）。')

    # —— 输出控制 ——
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='输出目录。')
    parser.add_argument('--output_plot', type=str, default='similarity_distribution.pdf',
                        help='（labeled）输出图表基础名。')
    parser.add_argument('--output_csv', type=str, default='similarity_stats.csv',
                        help='（labeled）输出统计 CSV 基础名。')
    parser.add_argument('--output_csv_unlabeled', type=str, default='unlabeled_similarity.csv',
                        help='（unlabeled）逐图结果 CSV 基础名。')

    # —— 模型/Loader ——
    parser.add_argument('--backbone_model', type=str, default='CLIP:ViT-B/32',
                        help='模型名称, 如 CLIP:ViT-B/32 或 DINOv2:dinov2_vitl14')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小。')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader 进程数。')

    # —— 变换类型与参数 ——
    parser.add_argument('--transform_type', type=str,
                        choices=['jpeg', 'blur', 'noise'], required=True,
                        help='变换类型 (jpeg / blur / noise)。')

    parser.add_argument('--jpeg_quality', type=int, nargs='+', default=[75],
                        help='一个或多个 JPEG 压缩质量 (1-100)。')
    parser.add_argument('--blur_sigma', type=float, nargs='+', default=[5.0],
                        help='一个或多个高斯模糊 sigma。')
    parser.add_argument('--blur_kernel', type=int, default=21, help='高斯模糊核大小。')

    parser.add_argument('--noise_std', type=float, nargs='+', default=[0.03, 0.05, 0.10],
                        help='一个或多个高斯噪声标准差(相对[0,1])。')

    # —— 无标签模式专属 ——
    parser.add_argument('--repeats', type=int, default=5,
                        help='unlabeled 模式下对同一参数做多少次重复采样（对噪声尤为有用）。')

    args = parser.parse_args()
    main(args)




    #单图命令
    # python
    # analyze_similarity.py \
    # - -transform_type
    # noise \
    # - -image_path / path / to / image.png \
    # - -unlabeled \
    # - -repeats
    # 10 \
    # - -noise_std
    # 0.02
    # 0.05
    # 0.10 \
    # - -output_dir
    # out_unlabeled_single

#目录命令
# python analyze_similarity.py \
#   --transform_type noise \
#   --data_dir /path/to/images \
#   --unlabeled \
#   --repeats 10 \
#   --noise_std 0.02 0.05 0.10 \
#   --output_dir out_unlabeled_dir


#前面改的小一点，我还能看懂一部分，后面几个是真看不懂了，目前该文件内没有参数命名冲突，其他文件我不知道
#绝大多数的部分没有完全重写，而是独立逻辑，能沿用的我就沿用了，具体内容上面写了在哪个问答下有更加具体的内容
#我去上课了，886
#ps:前面有整个文档详细说明，在最上面
#ps2: 几乎所有内容修改记录注释全部都有，依赖我没太搞清楚