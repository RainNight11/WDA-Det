# visA.py 修改版 -单独保存每张图
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import sys
import os
import torch.nn.functional as F

#避免CUDA导入问题
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 或 ""强制CPU

# 导入你的模型
sys.path.append('/data_B/tianyu/code/ICME-Attention')
from models.wda_model import WDAModel

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化"""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def save_individual_images(x, x_tilde, r, att_logits, A, x_att, pred, save_dir='./vis'):
    """单独保存每张图像"""
    os.makedirs(save_dir, exist_ok=True)

    # 移到CPU并转为numpy
    x_np = denormalize(x).cpu().squeeze().numpy()
    x_tilde_np = denormalize(x_tilde).cpu().squeeze().numpy()
    r_np = r.cpu().squeeze().numpy()
    att_logits_np = att_logits.cpu().squeeze().numpy()
    A_np = A.cpu().squeeze().numpy()
    x_att_np = denormalize(x_att).cpu().squeeze().numpy()

    # 转换维度 [C,H,W] -> [H,W,C]
    x_np = np.transpose(x_np, (1, 2, 0))
    x_tilde_np = np.transpose(x_tilde_np, (1, 2, 0))
    r_np = np.transpose(r_np, (1, 2, 0))
    x_att_np = np.transpose(x_att_np, (1, 2, 0))

    # 裁剪到[0,1]
    x_np = np.clip(x_np, 0, 1)
    x_tilde_np = np.clip(x_tilde_np, 0, 1)
    x_att_np = np.clip(x_att_np, 0, 1)

    # 计算辅助可视化图
    r_abs = np.abs(r_np)
    r_avg = r_abs.mean(axis=2)
    r_heatmap = np.mean(np.abs(r_np), axis=2)

    # === 单独保存每张图 ===

    # 1. 原始输入    plt.figure(figsize=(8, 8))
    plt.imshow(x_np)
    # plt.title('01_original_input')
    plt.axis('off')
    plt.savefig(f'{save_dir}/01_original_input.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 去噪图像 (x_tilde)
    plt.figure(figsize=(8, 8))
    plt.imshow(x_tilde_np)
    # plt.title('02_denoised_x_tilde')
    plt.axis('off')
    plt.savefig(f'{save_dir}/02_denoised_x_tilde.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 残差平均值 (r_abs avg)
    plt.figure(figsize=(8, 8))
    plt.imshow(r_avg, cmap='hot')
    # plt.title('03_residual_absolute_average')
    plt.colorbar(label='|r| (avg)')
    plt.axis('off')
    plt.savefig(f'{save_dir}/03_residual_absolute_average.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 残差RGB (r)
    plt.figure(figsize=(8, 8))
    plt.imshow(r_np)
    # plt.title('04_residual_rgb')
    plt.axis('off')
    plt.savefig(f'{save_dir}/04_residual_rgb.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 残差热力图 (viridis)
    plt.figure(figsize=(8, 8))
    plt.imshow(r_heatmap, cmap='viridis')
    # plt.title('05_residual_heatmap_viridis')
    plt.colorbar(label='|r| (avg)')
    plt.axis('off')
    plt.savefig(f'{save_dir}/05_residual_heatmap_viridis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. 残差热力图 (hot)
    plt.figure(figsize=(8, 8))
    plt.imshow(r_heatmap, cmap='hot')
    # plt.title('06_residual_heatmap_hot')
    plt.colorbar(label='|r| (avg)')
    plt.axis('off')
    plt.savefig(f'{save_dir}/06_residual_heatmap_hot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. 注意力Logits (sigmoid前)
    plt.figure(figsize=(8, 8))
    plt.imshow(att_logits_np, cmap='coolwarm')
    # plt.title('07_attention_logits')
    plt.colorbar(label='Logits')
    plt.axis('off')
    plt.savefig(f'{save_dir}/07_attention_logits.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 8. 注意力权重 (A, sigmoid后)
    plt.figure(figsize=(8, 8))
    plt.imshow(A_np, cmap='hot')
    # plt.title('08_attention_weights_A')
    plt.colorbar(label='Weight')
    plt.axis('off')
    plt.savefig(f'{save_dir}/08_attention_weights_A.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 9. 增强图像 (x_att)
    plt.figure(figsize=(8, 8))
    plt.imshow(x_att_np)
    # plt.title('09_enhanced_image')
    plt.axis('off')
    plt.savefig(f'{save_dir}/09_enhanced_image.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 10. 原始 vs 去噪 vs 增强 (三图对比)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(x_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(x_tilde_np)
    axes[1].set_title('Denoised')
    axes[1].axis('off')

    axes[2].imshow(x_att_np)
    axes[2].set_title('Enhanced')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_comparison_original_denoised_enhanced.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 11. 残差 vs 注意力权重 (关键对比)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(r_avg, cmap='hot')
    axes[0].set_title('Residual (Input to Attention)')
    axes[0].axis('off')

    axes[1].imshow(A_np, cmap='hot')
    axes[1].set_title('Attention Weights (Output)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/11_residual_vs_attention.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 12. 残差的三个通道分别可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    channel_names = ['Red', 'Green', 'Blue']
    for i in range(3):
        axes[i].imshow(r_np[:, :, i], cmap='coolwarm')
        axes[i].set_title(f'Residual Channel {channel_names[i]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/12_residual_channels.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 13. 注意力权重分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(A_np.flatten(), bins=50, color='darkred', alpha=0.7, edgecolor='black')
    plt.title('13_attention_weight_distribution')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/13_attention_weight_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 14. 残差值分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(r_avg.flatten(), bins=50, color='darkblue', alpha=0.7, edgecolor='black')
    plt.title('14_residual_value_distribution')
    plt.xlabel('Residual |r| (avg)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/14_residual_value_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 15. 统计信息汇总图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    stats_text = f"""
    RFNT Model Statistics

    Prediction: {pred.detach().cpu().numpy()[0]}

    Residual (r) Statistics:
    - Min: {r.min():.4f}
    - Max: {r.max():.4f}
    - Mean: {r.mean():.4f}
    - Std: {r.std():.4f}

    Attention Weights (A) Statistics:
    - Min: {A.min():.4f}
    - Max: {A.max():.4f}
    - Mean: {A.mean():.4f}
    - Std: {A.std():.4f}

    Enhanced Image (x_att) Statistics:
    - Min: {x_att.min():.4f}
    - Max: {x_att.max():.4f}
    - Mean: {x_att.mean():.4f}
    - Std: {x_att.std():.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.title('15_statistics_summary', fontsize=14, pad=20)
    plt.savefig(f'{save_dir}/15_statistics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 所有图像已保存到: {save_dir}/")
    print(f"  共生成15 张独立图像")
    print(f"  文件名格式: 数字_描述.png (便于排序)")

# 主程序
if __name__ == "__main__":
    # 1. 加载模型
    model = WDAModel(
        backbone_name="CLIP:ViT-L/14",
        num_classes=2,
        alpha=1.0,
        smooth=True,
        npr_layers=(2, 2),
        npr_use_abs=True,
        npr_scale_residual=True
    )

    # 加载权重（如果有）
    # checkpoint = torch.load("your_weights.pth", map_location="cpu")
    # model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. 准备输入
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用真实图像
    img_path = "./vis/0060.png"
    if not os.path.exists(img_path):
        print(f"警告: {img_path} 不存在，使用随机数据")
        x = torch.randn(1, 3, 224, 224).to(device)
    else:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

    # 3. 获取所有中间结果
    with torch.no_grad():
        # 获取去噪图像和残差
        x_tilde = model.denoise(x)
        r = x - x_tilde

        # 获取注意力logits（手动执行res_att前向）
        r_input = r.abs() if model.res_att.use_abs else r
        if model.res_att.scale_residual:
            r_input = r_input * (2.0 / 3.0)

        #执行res_att的前几层
        att_x = model.res_att.relu(model.res_att.bn1(model.res_att.conv1(r_input)))
        att_x = model.res_att.maxpool(att_x)
        att_x = model.res_att.layer1(att_x)
        att_x = model.res_att.layer2(att_x)

        # 获取logits和权重
        att_logits = model.res_att.att_head(att_x)
        A = torch.sigmoid(att_logits)
        A_upsampled = F.interpolate(A, size=x.shape[2:], mode="bilinear", align_corners=False)
        if model.res_att.smooth:
            A_upsampled = model.res_att.smoother(A_upsampled).clamp(0.0, 1.0)

        # 增强图像
        x_att = x_tilde * (1.0 + model.alpha * A_upsampled)

        # 预测
        feat = model.backbone_extract(x_att)
        feat_bn = model.bn(feat)
        pred = model.ClassifyNet(feat_bn)

    # 4. 单独保存每张图
    save_individual_images(x, x_tilde, r, att_logits, A_upsampled, x_att, pred)

    print("\n=== 完成 ===")
    print(f"预测结果: {pred.detach().cpu().numpy()[0]}")
    print(f"残差范围: [{r.min():.4f}, {r.max():.4f}]")
    print(f"注意力范围: [{A_upsampled.min():.4f}, {A_upsampled.max():.4f}]")
