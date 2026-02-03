from .clip import clip
from PIL import Image
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import torchvision.utils as vutils
CHANNELS = {
    # CLIP:
    "RN50" : 1024,
    "ViT-L/14" : 768,
    # DINOv2
    'DINOv2:ViT-S14':384,
    'DINOv2:ViT-B14':768,
    'DINOv2:ViT-L14':1024,
    'DINOv2:ViT-G14':1536,
}

# # CNN噪声生成器
# class NoiseGenerator(nn.Module):
#     def __init__(self, num_channels=3, noise_channels=3):
#         super(NoiseGenerator, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),  # 第1层卷积
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 第2层卷积，下采样
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 第3层卷积
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 第4层卷积
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 上采样
#             nn.ReLU(),
#             nn.Conv2d(64, noise_channels, kernel_size=3, stride=1, padding=1),  # 输出噪声
#             nn.Tanh(),
#         )
#     def forward(self, x):
#         encoded = self.encoder(x)
#         middle = self.middle(encoded)
#         noise = self.decoder(middle)
#         return noise

def icnr(tensor, scale=2, init=nn.init.kaiming_normal_):
    """
    ICNR 初始化：把上采样卷积权重初始化成最近邻插值的权重分布，
    参考 Aitken et al. 'Checkerboard artifact free sub-pixel convolution'
    """
    out_c, in_c, k, k2 = tensor.shape
    sub = torch.zeros(out_c // (scale ** 2), in_c, k, k2)
    init(sub)  # 正常的 Kaiming 初始化
    sub = sub.repeat_interleave(scale ** 2, 0)  # 按最近邻规则复制
    with torch.no_grad():
        tensor.copy_(sub)

"""
避免棋盘伪影
"""
# CNN噪声生成器
class NoiseGenerator(nn.Module):
    def __init__(self, num_channels=3, noise_channels=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 32, 3, padding=1, dilation=2)

        self.trans1 = torch.nn.ConvTranspose2d(32, 128, 3, padding=1, dilation=2)
        self.trans2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(32, 3, 3, padding=1)
        self.mp = torch.nn.MaxPool2d(2, return_indices=True)
        self.up = torch.nn.MaxUnpool2d(2)
        self.relu = torch.nn.ReLU()

    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # [?, 32, 224, 224]
        s1 = x.size()
        x, ind1 = self.mp(x)  # [?, 32, 112, 112]
        x = self.conv2(x)
        x = self.relu(x)  # [?, 64, 112, 112]
        s2 = x.size()
        x, ind2 = self.mp(x)  # [?, 64, 56, 56]
        x = self.conv3(x)
        x = self.relu(x)  # [?, 128, 56, 56]
        s3 = x.size()
        x, ind3 = self.mp(x)  # [?, 128, 28, 28]
        x = self.conv4(x)
        x = self.relu(x)  # [?, 32, 28, 28]

        return x, ind1, s1, ind2, s2, ind3, s3

    def decoder(self, x, ind1, s1, ind2, s2, ind3, s3):
        x = self.trans1(x)
        x = self.relu(x)  # [128, 128, 28, 28]
        x = self.up(x, ind3, output_size=s3)  # [128, 128, 56, 56]
        x = self.trans2(x)
        x = self.relu(x)  # [128, 128, 56, 56]
        x = self.up(x, ind2, output_size=s2)  # [128, 128, 112, 112]
        x = self.trans3(x)
        x = self.relu(x)  # [128, 128, 112, 112]
        x = self.up(x, ind1, output_size=s1)  # [128, 128, 224, 224]
        x = self.trans4(x)
        x = self.relu(x)  # [128, 128, 224, 224]
        return x

    def forward(self, x):
        x_in = x
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        output = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)

        # # -------- 仅用于可视化的加噪图像保存，不影响模型功能 --------
        # with torch.no_grad():
        #     vis_noise = output
        #
        #     # 如果噪声的空间尺寸和输入不一样，就插值到和 x_in 一样大
        #     if vis_noise.shape[-2:] != x_in.shape[-2:]:
        #         vis_noise = F.interpolate(
        #             vis_noise,
        #             size=x_in.shape[-2:],   # (H, W) = 输入图像的尺寸
        #             mode="bilinear",
        #             align_corners=False
        #         )
        #
        #     noisy_x = (x_in + vis_noise).clamp(0, 1)  # 加噪后的图像，截断到 [0,1]
        #     vutils.save_image(noisy_x, "noisy_debug.png")  # 这里你也可以换成别的路径
        # # --------------------------------------------------


        return output



# CNN噪声应用器
# class NoiseApplier(nn.Module):
#     def __init__(self, noise_generator):
#         super(NoiseApplier, self).__init__()
#         self.noise_generator = noise_generator
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         noise = torch.randn(batch_size, x.size(1),x.size(2),x.size(3), device=x.device)
#         noise = self.noise_generator(noise)
#         return x + noise

class NPRPerturb(nn.Module):
    def __init__(self, p_nearest=0.5, p_bilinear=0.5, gauss_sigma=1.0/255.0, use_checker=True):
        super().__init__()
        self.p_nearest = p_nearest
        self.p_bilinear = p_bilinear
        self.gauss_sigma = gauss_sigma
        self.use_checker = use_checker

    @staticmethod
    def downup(x, s=0.5, mode='nearest'):
        y = F.interpolate(F.interpolate(x, scale_factor=s, mode=mode, recompute_scale_factor=True),
                          scale_factor=1/s, mode=mode, recompute_scale_factor=True)
        return y

    def checkerboard(self, x):
        # 2×2 棋盘式保留：保留每 2×2 的(0,0)像素，其余位置轻度插值
        B, C, H, W = x.shape
        mask = torch.zeros((1,1,H,W), device=x.device)
        mask[:, :, 0::2, 0::2] = 1.0
        # 对“被掩”的位置做轻度平滑近似插值（3×3 均值核）
        kernel = torch.ones((C,1,3,3), device=x.device) / 9.0
        smooth = F.conv2d(x, kernel, padding=1, groups=C)
        return x * mask + smooth * (1-mask)

    def forward(self, npr_feat):
        z = npr_feat
        # ① 最近邻/双线性下上采样（与上采样伪迹同构）
        if torch.rand(1, device=z.device) < self.p_nearest:
            z = z - self.downup(z, 0.5, mode='nearest')
        if torch.rand(1, device=z.device) < self.p_bilinear:
            z = z - self.downup(z, 0.5, mode='bilinear')

        # ② 棋盘式亚采样修复（强化 2×2 结构）
        if self.use_checker:
            z = self.checkerboard(z)

        # ③ 低强度高斯（最后加一点轻噪声，保证“微扰且语义稳”）
        if self.gauss_sigma > 0:
            z = z + torch.randn_like(z) * self.gauss_sigma
        return z




class NPRLayer(nn.Module):
    def __init__(self, scales=(0.5,)):
        super().__init__()
        self.scales = scales
    def forward(self, x):
        outs = []
        for s in self.scales:
            y = F.interpolate(F.interpolate(x, scale_factor=s, mode='nearest', recompute_scale_factor=True),
                              scale_factor=1/s, mode='nearest', recompute_scale_factor=True)
            outs.append(x - y)  # 2×2 相对差的近似
        return torch.cat(outs, dim=1)  # [B, 3*len(scales), H, W]


class RFNTModel(nn.Module):
    """
    loader size: every_loader图片张数 * 3 * 224 * 224
    """
    def __init__(self, backbone_name, num_classes=1):
        super(RFNTModel, self).__init__()
        self.bk_name = backbone_name
        """该部分为一种npr增强"""
        self.npr = NPRLayer(scales=(0.5, 0.25))
        self.npr_perturb = NPRPerturb(p_nearest=1.0, p_bilinear=0.5, gauss_sigma=1 / 255.0, use_checker=True)
        self.proj_to3 = nn.Conv2d(3*2, 3, kernel_size=1)  # 因为 scales=(.5,.25) → 通道=3*2


        if backbone_name.startswith("DINOv2"):
            """
            DINOv2:ViT-L14
            """
            # backbone_name = backbone_name[7:]
            # 取冒号后面的部分，例如 "ViT-L14"
            model_tag = backbone_name.split(":")[1]           # "ViT-L14"
            # 统一转换成小写并去掉中间的连字符，例如："vitl14"
            model_tag = model_tag.replace("-", "").lower()    # "vitl14"
            # 拼出 torch.hub 所需的格式
            hub_name = f"dinov2_{model_tag}"                 # "dinov2_vitl14"

            # 加载模型
            self.pre_model = torch.hub.load(
                'models/dinov2',
                hub_name,
                source='local'
            ).cuda()
            # self.pre_model = torch.hub.load('models/dinov2', 'dinov2_vitl14',source='local').cuda()
        elif backbone_name.startswith("CLIP"):
            backbone_name = backbone_name[5:]
            self.pre_model, self.preprocess = clip.load(backbone_name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        self.bn = nn.BatchNorm1d(CHANNELS[backbone_name])  # 使用BatchNorm2d层
        self.ClassifyNet = nn.Sequential(
            # 全连接层 1: 输入 768，输出 256
            nn.Linear(CHANNELS[backbone_name], num_classes),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        # 初始化 CNN 噪声生成器
        self.noise_generator = NoiseGenerator(num_channels=3, noise_channels=3)

    def add_noise(self, x):
        x = self.noise_generator(x)
        return x


    """
    这一部分是关于相似度的计算
    """
    # DCA相似我的写法
    def dca_similarity(self,x1, x2):
        # version2:协方差
        batch_size = x1.shape[0]

        # 对 x1 和 x2 分别计算均值
        x1_mean = torch.mean(x1, dim=1, keepdim=True)  # 每个样本的均值
        x2_mean = torch.mean(x2, dim=1, keepdim=True)

        # 将每个样本居中
        x1_centered = x1 - x1_mean
        x2_centered = x2 - x2_mean

        # 计算标准差
        x1_std = torch.std(x1, dim=1, unbiased=True, keepdim=True)
        x2_std = torch.std(x2, dim=1, unbiased=True, keepdim=True)

        # 计算协方差矩阵： batch_size 个协方差矩阵，每个为 768x768
        cov_matrices = []
        for i in range(batch_size):
            # 计算协方差矩阵
            cov_matrix = torch.mm(x1_centered[i].unsqueeze(1), x2_centered[i].unsqueeze(0)) / (x1.shape[1] - 1)
            # 标准化：除以对应的 x1_std * x2_std
            std_product = x1_std[i] * x2_std[i]  # 对应 batch 的 std 乘积
            cov_matrix_normalized = cov_matrix / std_product  # 元素级除法
            cov_matrices.append(cov_matrix_normalized)

        # 将 cov_matrices 列表转换为 batch_size x 768 x 768 的张量
        output = torch.stack(cov_matrices, dim=0)
        # 提取每个 batch 中 768x768 矩阵的对角线
        # output = torch.diagonal(output, dim1=-2, dim2=-1)
        return output

    # DCA相似，GPT写法
    def diag_corr(self, x1, x2, eps=1e-6):
        # x1,x2: [B, D]
        x1m = x1 - x1.mean(dim=1, keepdim=True)
        x2m = x2 - x2.mean(dim=1, keepdim=True)
        x1s = x1m / (x1m.std(dim=1, keepdim=True) + eps)
        x2s = x2m / (x2m.std(dim=1, keepdim=True) + eps)
        # per-dim 相关向量 r ∈ R^D
        r = (x1s * x2s)  # [B, D]
        # 也可把 (1-r) 看成“漂移”，再接 BN/MLP
        return r

    def similarity_compute(self,x1,x2,mode = 'dca_similarity'):
        if mode.startswith("dca_similarity"):
            return self.dca_similarity(x1,x2)
        elif mode.startswith("diag_corr"):
            return self.diag_corr(x1,x2)


    """以下是NPR特征，除非使用了NPR增强，否则一般不用"""
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)


    """该部分是backbone的选择"""
    def backbone_extract(self,x):
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)              # [B, D]
        else:
            return self.pre_model.encode_image(x) # [B, D]



    def forward(self, x, return_feature=False):
        # 使用NPR增强
        # x = self.interpolate(x,0.5)
        # 对输入图像扰动
        x_noised = self.add_noise(x)
        # 特征提取
        org_feature = self.backbone_extract(x)
        noise_feature = self.backbone_extract(x_noised)
        # 相似度计算
        # dca
        # sim = self.similarity_compute(org_feature,noise_feature,"dca_similarity")  # dca or diag_corr
        # sim = self.pooling(sim)
        # sim = sim.squeeze(-1)

        ## diag_corr
        sim = self.similarity_compute(org_feature,noise_feature,"diag_corr")  # dca or diag_corr

        ##########
        sim = self.bn(sim)

        if return_feature: # 这个是用来调整画图模式的，可以不动他
            return sim

        pred = self.ClassifyNet(sim)  # [B,1] or [B,num_classes]
        return pred, org_feature - noise_feature
