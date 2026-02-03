from .clip import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
# moe_models.py
from models.ddim_models import DDIM
from models.sd_models import SDModel

class MoEGate(nn.Module):
    def __init__(self, input_channels, hidden_dim=512):
        super(MoEGate, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2个专家：DDIM和SD

    def forward(self, x):
        # 假设输入 x 是经过卷积层提取的特征图
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)  # 输出DDIM和SD的选择概率

class FeatureExtractor(nn.Module):  # need to modify
    def __init__(self, in_channels=3, feature_dim=512):
        super(FeatureExtractor, self).__init__()
        # 使用 ResNet50 作为特征提取器
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉最后的分类层

        # 用一个卷积层或全连接层进行特征降维
        self.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)
    
    def forward(self, x):
        features = self.resnet(x)  # 提取特征
        features = self.fc(features)  # 将特征降维到合适的大小
        return features

class MoEDiffusionModel(nn.Module):
    def __init__(self, opt):
        super(MoEDiffusionModel, self).__init__()
        
        # 加载DDIM和SD模型
        self.ddim_expert = DDIM(opt)  # 假设opt中包含模型的配置
        self.sd_expert = SDModel(opt)  # 同上
        
        # 特征提取网络，先对图像进行降维
        self.feature_extractor = FeatureExtractor(in_channels=3, feature_dim=512)
        
        # 门控网络
        self.gate = MoEGate(512)  # 输入大小需要与特征提取后的维度一致
    
    def forward(self, x):
        # 使用特征提取网络提取图像特征
        feature = self.feature_extractor(x)
        
        # 使用门控网络决定使用哪个专家（DDIM 或 SD）
        gate_output = self.gate(feature)
        ddim_prob = gate_output[:, 0]
        sd_prob = gate_output[:, 1]
        
        # 选择专家进行推理
        ddim_output = self.ddim_expert(x) * ddim_prob.unsqueeze(-1).unsqueeze(-1)
        sd_output = self.sd_expert(x) * sd_prob.unsqueeze(-1).unsqueeze(-1)
        
        # 根据概率组合输出
        output = ddim_output + sd_output
        return output

class MoEDDModel(nn.Module):
    def __init__(self, opt):
        super(MoEDDModel, self).__init__()
        self.moediffusion = MoEDiffusionModel(opt)  
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 二分类：真或伪

    def forward(self, x):
        # choose reconst
        reconstructed_image = self.moediffusion(x)
        
        reconstructed_image = F.interpolate(reconstructed_image, size=(224, 224))
        
        # resnet classify
        output = self.resnet(reconstructed_image)
        return output
