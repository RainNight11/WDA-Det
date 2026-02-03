import functools
import os

import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt.arch)
        # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        """增加多卡训练"""
        # 检查 GPU 数量并选择使用单卡或多卡模式
        if len(opt.gpu_ids) > 1:
            # 使用多卡模式
            self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpu_ids)
            self.device = torch.device(f'cuda:{opt.gpu_ids[0]}')  # 主设备是第一个 GPU
        elif len(opt.gpu_ids) == 1:
            # 使用单卡模式
            self.device = torch.device(f'cuda:{opt.gpu_ids[0]}')
        else:
            # 使用 CPU
            self.device = torch.device('cpu')

        # 将模型移动到指定的设备上
        self.model.to(self.device)

        ### 对于不同模型初始化
        if opt.arch.startswith("RFNT"):
            # 训练的层
            param_names = [
                "bn",
                "ClassifyNet",
                "noise_generator",
                "pooling"
            ]

        elif opt.arch.startswith("CLIP:"):
            param_names = ['fc']


        if opt.fix_backbone:
            # 定义需要设置 requires_grad = True 的参数名称列表
            # # 如果使用了 DataParallel，就给参数名称添加 'module.' 前缀
            # if isinstance(self.model, torch.nn.DataParallel):
            #     param_names = [f"module.{name}" for name in param_names]
            #
            # params = []
            # for name, p in self.model.named_parameters():
            #     if name in param_names:
            #         p.requires_grad = True
            #         params.append(p)
            #     else:
            #         p.requires_grad = False
            # 如果使用了 DataParallel，则在 param_names 前加 'module.'
            if isinstance(self.model, torch.nn.DataParallel):
                param_names = [f"module.{name}" for name in param_names]

            params = []
            for name, p in self.model.named_parameters():
                # 检查参数名称是否以 `param_names` 中的某个前缀开始
                if any(name.startswith(prefix) for prefix in param_names):
                    p.requires_grad = True
                    params.append(p)
                else:
                    p.requires_grad = False

        else:
            print(
                "Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999),
                                               weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")
        # 如果 new_optim 为 False，则尝试加载优化器状态

        if not opt.new_optim :
            self.load_networks()  # 加载模型和优化器的状态

        self.loss_fn = nn.BCEWithLogitsLoss()


    def load_networks(self):
        load_path = self.opt.lastload_path
        if self.opt.last_epoch != -1 and os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            # # 移除 `module.` 前缀
            # new_state_dict = {}
            # for key, value in checkpoint.items():
            #     new_key = key.replace("module.", "")
            #     new_state_dict[new_key] = value

            self.model.load_state_dict(checkpoint['model'])
            # 仅当 new_optim 为 False 时才加载优化器状态
            if not self.opt.new_optim:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.total_steps = checkpoint['total_steps']
            print(f"Loaded checkpoint '{load_path}' with total steps: {self.total_steps}")
        else:
            print(f"No checkpoint found at '{load_path}'")

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        # self.output, self.cos_sim = self.model(self.input)
        if self.opt.arch.startswith("RFNT"):
            self.output,self.process_feature= self.model(self.input)
        elif self.opt.arch.startswith("CLIP:"):
            param_names = ['fc']
            self.output= self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_covariance_loss(self, cos_sim, label):
        """
        计算自定义的协方差损失 (Loss_cov)
        :param cos_sim: 模型输出的余弦相似度 (batch_size, 1, 768)
        :param label: 标签 (batch_size,)
        :return: Loss_cov 损失值
        """
        batch_size, dim = cos_sim.size()  # 获取批次大小和第三维度 (768)

        # 确保 label 的维度是 (batch_size,) 并扩展到 (batch_size, 1, 768)
        label_expanded = label.view(batch_size, 1, 1).expand(batch_size, 1, dim)  # (batch_size, 1, 768)

        # 计算 (cos_sim - (1 - label))^2
        loss_elementwise = (cos_sim - (1 - label_expanded.float())) ** 2
        # loss_elementwise = (cos_sim - label_expanded.float()) ** 2

        # 对第三维度求和 (768 个元素)
        loss_per_sample = loss_elementwise.sum(dim=2)  # (batch_size, 1)

        # 对 batch_size 维度求和，计算总损失
        total_loss = loss_per_sample.sum()  # 标量

        # 最后除以 batch_size 和 768
        loss_cov = total_loss / (batch_size * dim)

        return loss_cov

    def real_consistency_loss(self, minus_feature, label):
    # label: [B], 0=real, 1=fake
        mask_real = (label == 0).float().unsqueeze(1)  # [B,1]
        delta = (minus_feature).pow(2).sum(dim=1, keepdim=True).sqrt()  # [B,1]
        loss_real = (delta * mask_real).sum() / (mask_real.sum() + 1e-6)
        return loss_real

    def get_loss(self):
        loss_bce = self.loss_fn(self.output.squeeze(1), self.label)
        if self.opt.arch.startswith("RFNT"):
            if self.opt.loss == "loss_t":
                    loss_cov = self.get_covariance_loss(self.process_feature, self.label)
                    return loss_cov + loss_bce
            elif self.opt.loss == "loss_real_consistency":
                loss_real = self.real_consistency_loss(self.process_feature, self.label)
                lambda_real = 0.1
                return loss_real + lambda_real*loss_bce
            elif self.opt.loss == "loss_bce":
                return loss_bce
        else:
            return loss_bce
    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



