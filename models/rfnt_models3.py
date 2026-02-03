from .clip import clip
from PIL import Image
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    'DINOv2-ViT-L14':1024,
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
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        output = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)
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


class RFNTModel(nn.Module):
    """
    loader size: every_loader图片张数 * 3 * 224 * 224
    """
    def __init__(self, backbone_name, num_classes=1):
        super(RFNTModel, self).__init__()
        self.bk_name = backbone_name
        if backbone_name.startswith("DINOv2"):
            self.pre_model = torch.hub.load('models/dinov2', 'dinov2_vitl14',source='local').cuda()
        else:
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

    def cosine_similarity(self, x1, x2):
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

    def forward_fft(self, x):
        # 应用傅立叶变换
        x_fft = torch.fft.fft2(x)
        # 取模并对数缩放
        x_fft = torch.log(torch.abs(x_fft) + 1e-10)
        # 归一化，将傅立叶变换结果缩放到 [-1, 1] 的范围
        x_min = x_fft.min()
        x_max = x_fft.max()
        x_fft = 2 * (x_fft - x_min) / (x_max - x_min) - 1
        return x_fft

    def forward_grad(self, x):
        # 计算 x 和 y 方向的 Sobel 梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3).repeat(3,1,1,1).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3).repeat(3,1,1,1).to(x.device)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=3)
        # 叠加梯度大小
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # 将梯度归一化到 [0, 1] 范围
        grad_min = grad.min()
        grad_max = grad.max()
        grad= (grad - grad_min) / (grad_max - grad_min + 1e-8)  # 加上小的 epsilon 防止除以0

        # 将归一化的梯度缩放到 [-1, 1] 范围
        grad = 2 * grad - 1
        return grad

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)


    def backbone_extract(self,x):
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)              # [B, D]
        else:
            return self.pre_model.encode_image(x) # [B, D]


    def forward(self, x, return_feature=False):
        x = x - self.interpolate(x, 0.5)  # 使用npr
        # """尝试直接使用rgb"""
        # # 加噪
        # # x_noised = self.add_noise(x)
        # # save_image(x_noised,os.path.join("visual","noise_image.png"))
        # # 使用clip进行特征提取
        # if self.bk_name.startswith("DINOv2"):
        #     org_feature = self.pre_model(x)
        #     noise_feature = self.pre_model(x_noised)  # （256，768
        # else:
        #     org_feature = self.pre_model.encode_image(x)
        #     noise_feature = self.pre_model.encode_image(x_noised) # （256，768
        # # 计算 协方差矩阵
        # cos_matrix = self.cosine_similarity(org_feature, noise_feature)
        # # 调用最大池化
        # cos_sim = self.pooling(cos_matrix)  # 平均池化到 (batch_size, 768)
        # # 去掉多余的维度
        # cos_sim = cos_sim.squeeze(-1)  # 输出形状为 (batch_size, 768)
        # # 确认cos_sim的形状
        # print("Shape of cos_sim: ", cos_sim.shape)
        # cos_sim = self.bn(cos_sim)
        # if return_feature:
        #     return cos_sim
        # cos_sim = cos_sim.unsqueeze(1)
        # pred = self.ClassifyNet(cos_sim)
        if self.bk_name.startswith("DINOv2"):
            org_feature = self.pre_model(x)
        else:
            org_feature = self.pre_model.encode_image(x)
        pred = self.ClassifyNet(org_feature)

        return pred, org_feature

#
# class RFNTModel(nn.Module):
#     """
#     loader size: every_loader图片张数 * 3 * 224 * 224
#
#     """
#
#     def __init__(self, name, num_classes=1):
#         super(RFNTModel, self).__init__()
#
#         self.pre_model, self.preprocess = clip.load("ViT-L/14",
#                                                     device="cpu")  # self.preprecess will not be used during training, which is handled in Dataset class
#         # self.fc = nn.Linear(160, num_classes )
#         # # 1x1卷积，将9个通道减少到3个通道
#         # self.conv1x1 = nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0)
#         self.bn = nn.BatchNorm1d(768)  # 使用BatchNorm2d层
#         # 冻结 conv1x1 的参数，使其不更新
#         # for param in self.conv1x1.parameters():
#         #     param.requires_grad = False
#         self.ClassifyNet = nn.Sequential(
#             # 全连接层 1: 输入 768，输出 256
#             nn.Linear(768, num_classes),
#             # nn.ReLU(),
#             # # 全连接层 2: 输出 2（二分类）
#             # nn.Linear(256, 128),
#             # nn.ReLU(),
#             # 全连接层 3: 输入 768，输出 256
#             # nn.Linear(256, num_classes)
#         )
#         # self.ClassifyNet = nn.Sequential(
#         #     # 卷积层 1: 输入 1 个通道（因为协方差矩阵是单通道的），输出 32 个特征图
#         #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 输出: (batch_size, 32, 768, 768)
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=2),  # 输出: (batch_size, 32, 384, 384)
#         #
#         #     # 卷积层 2: 输出 64 个特征图
#         #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输出: (batch_size, 64, 384, 384)
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=2),  # 输出: (batch_size, 64, 192, 192)
#         #
#         #     # 卷积层 3: 输出 128 个特征图
#         #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输出: (batch_size, 128, 192, 192)
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=2),  # 输出: (batch_size, 128, 96, 96)
#         #
#         #     # 卷积层 4: 输出 256 个特征图
#         #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 输出: (batch_size, 256, 96, 96)
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=2),  # 输出: (batch_size, 256, 48, 48)
#         #
#         #     # 将输出展平
#         #     nn.Flatten(),
#         #
#         #     # 全连接层 1: 输入 (256 * 48 * 48)，输出 256
#         #     nn.Linear(256 * 48 * 48, 256),
#         #     nn.ReLU(),
#         #
#         #     # 全连接层 2: 输出 num_classes
#         #     nn.Linear(256, num_classes)
#         # )
#         # 初始化 CNN 噪声生成器
#         self.noise_generator = NoiseApplier(NoiseGenerator(num_channels=3, noise_channels=3))
#
#     # def forward(self, x, return_feature=False):
#     #     # 原始图像
#     #     x_rgb = x
#     #     # FFT 和梯度
#     #     x_fft = self.forward_fft(x)
#     #     x_grad = self.forward_grad(x)
#     #     # 在通道维度上堆叠
#     #     x_combined = torch.cat([x_rgb, x_fft, x_grad], dim=1)
#     #     # 加噪
#     #     x_noised = self.add_noise(x_combined)
#     #     # 使用卷积融合
#     #     x_combined = self.conv1x1(x_combined)
#     #     x_noised = self.conv1x1(x_noised)
#     #     # 使用clip进行特征提取
#     #     org_feature = self.pre_model.encode_image(x_combined)
#     #     noise_feature = self.pre_model.encode_image(x_noised) # （256，768
#     #     # 计算余弦相似度
#     #     # 余弦相似度
#     #     cos_sim = self.cosine_similarity(org_feature, noise_feature)
#     #     # 确认cos_sim的形状
#     #     print("Shape of cos_sim: ", cos_sim.shape)
#     #     if return_feature:
#     #         return cos_sim
#     #     pred = self.ClassifyNet(cos_sim)
#     #     return pred
#
#     def add_noise(self, x):
#         # 加入高斯噪声
#         # noise = torch.randn_like(x) * 0.1  # 标准差为 0.1
#         # return x + noise
#         # 使用cnnNoiseApplier来生成并加噪
#         return self.noise_generator(x)
#
#     def cosine_similarity(self, x1, x2):
#         # batch_size = x1.shape[0]
#         # # 展平x1,x2
#         # x1 = x1.view(batch_size, -1)
#         # x2 = x2.view(batch_size, -1)
#         # # 计算余弦相似度
#         # cos_sim = F.cosine_similarity(x1, x2, dim=1)
#         # cos_sim = cos_sim.view(batch_size, 1)
#
#         # batch_size = x1.shape[0]
#         # x1 = F.normalize(x1, p=2, dim=1)
#         # x2 = F.normalize(x2, p=2, dim=1)
#         # x1 = x1.view(batch_size, -1)
#         # x2 = x2.view(batch_size, -1)
#         #
#         # # 使用 bmm 进行批量矩阵乘法
#         # output = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)).squeeze()  # 使用 bmm 进行批量矩阵乘法
#         # output = torch.mean(input, dim=2)  # 对最后一个维度进行平均值计算
#         # # version1:x1-x2
#         # output = x1 - x2  # 256,768
#
#         # version2:协方差
#         batch_size = x1.shape[0]
#
#         # 对 x1 和 x2 分别计算均值
#         x1_mean = torch.mean(x1, dim=1, keepdim=True)  # 每个样本的均值
#         x2_mean = torch.mean(x2, dim=1, keepdim=True)
#
#         # 将每个样本居中
#         x1_centered = x1 - x1_mean
#         x2_centered = x2 - x2_mean
#
#         # 计算标准差
#         x1_std = torch.std(x1, dim=1, unbiased=True, keepdim=True)
#         x2_std = torch.std(x2, dim=1, unbiased=True, keepdim=True)
#
#         # 计算协方差矩阵： batch_size 个协方差矩阵，每个为 768x768
#         cov_matrices = []
#         for i in range(batch_size):
#             # 计算协方差矩阵
#             cov_matrix = torch.mm(x1_centered[i].unsqueeze(1), x2_centered[i].unsqueeze(0)) / (x1.shape[1] - 1)
#             # 标准化：除以对应的 x1_std * x2_std
#             std_product = x1_std[i] * x2_std[i]  # 对应 batch 的 std 乘积
#             cov_matrix_normalized = cov_matrix / std_product  # 元素级除法
#             cov_matrices.append(cov_matrix_normalized)
#
#         # 将 cov_matrices 列表转换为 batch_size x 768 x 768 的张量
#         output = torch.stack(cov_matrices, dim=0)
#
#         # # 提取每个 batch 的协方差矩阵对角线上的元素
#         # diagonals = output.diagonal(dim1=-2, dim2=-1)
#
#         # 获取最大和最小值
#         max_value = output.max().item()
#         min_value = output.min().item()
#         # print(f"Max value: {max_value}, Min value: {min_value}")
#         # 提取每个 batch 中 768x768 矩阵的对角线
#         output = torch.diagonal(output, dim1=-2, dim2=-1)
#
#         # 确保 cos_sim_diag 的形状为 (batch_size, 768)
#         # output = output.permute(0, 2, 1)  # 将对角线特征调整为 (batch_size, 768)
#
#         return output
#
#     def forward_fft(self, x):
#         # 应用傅立叶变换
#         x_fft = torch.fft.fft2(x)
#         # 取模并对数缩放
#         x_fft = torch.log(torch.abs(x_fft) + 1e-10)
#         # 归一化，将傅立叶变换结果缩放到 [-1, 1] 的范围
#         x_min = x_fft.min()
#         x_max = x_fft.max()
#         x_fft = 2 * (x_fft - x_min) / (x_max - x_min) - 1
#         return x_fft
#
#     def forward_grad(self, x):
#         # 计算 x 和 y 方向的 Sobel 梯度
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1,
#                                                                                                                   1,
#                                                                                                                   1).to(
#             x.device)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1,
#                                                                                                                   1,
#                                                                                                                   1).to(
#             x.device)
#         grad_x = F.conv2d(x, sobel_x, padding=1, groups=3)
#         grad_y = F.conv2d(x, sobel_y, padding=1, groups=3)
#         # 叠加梯度大小
#         grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
#
#         # 将梯度归一化到 [0, 1] 范围
#         grad_min = grad.min()
#         grad_max = grad.max()
#         grad = (grad - grad_min) / (grad_max - grad_min + 1e-8)  # 加上小的 epsilon 防止除以0
#
#         # 将归一化的梯度缩放到 [-1, 1] 范围
#         grad = 2 * grad - 1
#         return grad
#
#     def forward(self, x, return_feature=False):
#         # 原始图像
#         # x_rgb = x
#         # # FFT 和梯度
#         # x_fft = self.forward_fft(x)
#         # x_grad = self.forward_grad(x)
#         # # 在通道维度上堆叠
#         # x_combined = torch.cat([x_rgb, x_fft, x_grad], dim=1)
#         # # 加噪
#         # x_noised = self.add_noise(x_combined)
#
#         """尝试直接使用rgb"""
#         # 加噪
#         x_noised = self.add_noise(x)
#         # 使用clip进行特征提取
#         org_feature = self.pre_model.encode_image(x)
#         noise_feature = self.pre_model.encode_image(x_noised)  # （256，768
#         # 计算余弦相似度
#         # 余弦相似度
#         cos_sim = self.cosine_similarity(org_feature, noise_feature)
#         # # 确认cos_sim的形状
#         # print("Shape of cos_sim: ", cos_sim.shape)
#         cos_sim = self.bn(cos_sim)
#         if return_feature:
#             return cos_sim
#         cos_sim = cos_sim.unsqueeze(1)
#         pred = self.ClassifyNet(cos_sim)
#         return pred, cos_sim
#
