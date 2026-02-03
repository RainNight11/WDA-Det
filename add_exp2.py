"""
analyze_adversarial_similarity.py (v2)

该脚本用于分析真实/伪造图像在遭受对抗性攻击后，其 CLIP 特征的变化。
此版本仿照 `validate_pgd.py` 的稳定实现，直接在特征空间中进行攻击，
目标是最大化原始特征与对抗特征之间的距离。

工作流程:
1.  加载数据集，区分真实 (real) 与伪造 (fake) 图像。
2.  实现基于特征距离的 FGSM, BIM, PGD 攻击算法。
3.  对每张图像，使用选定的攻击方法生成对抗样本。
4.  计算原始图像和对抗图像在 CLIP 特征空间中的余弦相似度。
5.  为真实图像和伪造图像分别统计和可视化相似度分布，并为每个攻击参数组合
    生成独立的图表和 CSV 报告。

此版本不再需要 `torchattacks` 库或类别标签。
"""
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
import pandas as pd

# 假设 models_v2.py 在正确的路径下
from models.models_v2 import FeatureMetric

class SimpleRealFakeDataset(Dataset):
    """
    一个简化的数据集类，仅加载图像并区分 real/fake。
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
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"错误：无法加载图像 {img_path}: {e}")
            return None, -1

def collate_fn_simple(batch):
    """自定义的 collate_fn，过滤掉加载失败的图像"""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


# --- 内联对抗性攻击实现 ---

def attack_feature_space(feature_model, images, eps, alpha, steps, attack_type):
    """
    在特征空间中执行对抗性攻击 (PGD, BIM, FGSM)。
    目标是最大化原始特征和对抗特征之间的余弦距离。
    """
    device = images.device
    original_images = images.detach().clone()

    with torch.no_grad():
        original_features = feature_model(original_images).detach()

    adv_images = images.detach().clone()
    # 仅 PGD 需要随机初始化
    if attack_type == 'pgd':
        adv_images += torch.zeros_like(images).uniform_(-eps, eps)

    num_steps = 1 if attack_type == 'fgsm' else steps

    for _ in range(num_steps):
        adv_images.requires_grad = True
        adv_features = feature_model(adv_images)

        # 损失是负的余弦相似度，最小化它等于最大化距离
        loss = -F.cosine_similarity(adv_features, original_features).sum()
        loss.backward()

        attack_grad = adv_images.grad.detach().sign()
        adv_images = adv_images.detach() + alpha * attack_grad

        delta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        adv_images = torch.clamp(original_images + delta, -2.5, 2.5) # Clamp to valid range

    return adv_images.detach()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 准备模型 ---
    print(f"正在加载 CLIP 模型: {args.clip_model}")
    feature_extractor = FeatureMetric(f"CLIP:{args.clip_model}").to(device).eval()

    # --- 准备数据 ---
    clip_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset = SimpleRealFakeDataset(root_dir=args.data_dir, transform=clip_preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_simple)

    # --- 开始分析 ---
    os.makedirs(args.output_dir, exist_ok=True)
    real_sims, fake_sims = [], []

    attack_name = args.attack.lower()
    alpha = args.alpha if args.alpha is not None else args.eps / args.steps

    print(f"\n开始使用 {attack_name.upper()} 进行分析 (eps={args.eps:.4f}, alpha={alpha:.4f}, steps={args.steps})...")

    for images, labels in tqdm(dataloader):
        if images is None:
            continue

        images = images.to(device)

        # 生成对抗样本
        adv_images = attack_feature_space(
            feature_model=feature_extractor,
            images=images,
            eps=args.eps,
            alpha=alpha,
            steps=args.steps,
            attack_type=attack_name
        )

        # 提取特征并计算相似度
        with torch.no_grad():
            feat_orig = feature_extractor(images)
            feat_adv = feature_extractor(adv_images)
            similarities = F.cosine_similarity(feat_orig, feat_adv, dim=1)

        # 收集结果
        for i in range(similarities.shape[0]):
            if labels[i] == 0:
                real_sims.append(similarities[i].item())
            elif labels[i] == 1:
                fake_sims.append(similarities[i].item())

    real_sims, fake_sims = np.array(real_sims), np.array(fake_sims)

    # --- 报告和保存结果 ---
    param_str = f"eps{args.eps:.3f}_steps{args.steps}"

    results = []
    if len(real_sims) > 0:
        results.append({
            'attack': attack_name, 'parameter': param_str, 'image_type': 'Real',
            'count': len(real_sims), 'mean_similarity': np.mean(real_sims), 'std_similarity': np.std(real_sims)
        })
    if len(fake_sims) > 0:
        results.append({
            'attack': attack_name, 'parameter': param_str, 'image_type': 'Fake',
            'count': len(fake_sims), 'mean_similarity': np.mean(fake_sims), 'std_similarity': np.std(fake_sims)
        })
    df = pd.DataFrame(results)

    print("\n--- 分析结果 ---")
    print(df.to_string(index=False, float_format="%.4f"))

    output_plot_path = os.path.join(args.output_dir, f"{args.attack}_{param_str}_similarity.pdf")
    output_csv_path = os.path.join(args.output_dir, f"{args.attack}_{param_str}_stats.csv")

    df.to_csv(output_csv_path, index=False)
    print(f"统计数据已保存至: {output_csv_path}")

    plt.figure(figsize=(12, 7))
    plt.hist(real_sims, bins=100, alpha=0.7, label='Real Images', density=True, range=(0.85, 1.0))
    plt.hist(fake_sims, bins=100, alpha=0.7, label='Fake Images', density=True, range=(0.85, 1.0))
    plt.title(f'Cosine Similarity Distribution (Attack: {attack_name.upper()}, Params: {param_str})', fontsize=16)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig(output_plot_path)
    plt.close()
    print(f"分析图表已保存至: {output_plot_path}\n")
    print("分析完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析 CLIP 特征在对抗性攻击下的余弦相似度 (v2)。')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集的根目录。')
    parser.add_argument('--output_dir', type=str, default='adversarial_results', help='存放所有输出文件的目录。')
    parser.add_argument('--attack', type=str.lower, required=True, choices=['fgsm', 'bim', 'pgd'], help='要使用的对抗性攻击方法。')

    # --- 攻击参数 ---
    parser.add_argument('--eps', type=float, default=8/255, help='对抗性扰动的最大范数 (epsilon)。')
    parser.add_argument('--steps', type=int, default=10, help='迭代攻击的步数 (用于 PGD, BIM)。FGSM 会忽略此参数。')
    parser.add_argument('--alpha', type=float, default=None, help='迭代攻击的步长。如果未提供，则默认为 eps/steps。')

    # --- 基础设置 ---
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='要使用的 CLIP 模型名称。')
    parser.add_argument('--batch_size', type=int, default=16, help='处理数据的批次大小。')
    parser.add_argument('--num_workers', type=int, default=16, help='Dataloader 使用的进程数。')

    args = parser.parse_args()
    main(args)

# python add_exp2.py --data_dir ../Datasets/progan_1percent/train/progan --output_dir adversarial_results/fgsm_analysis --attack fgsm --eps 0.03 --steps 10 --batch_size 8
