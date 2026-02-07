import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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
                "ClassifyNet",
                "noise_generator",
                "pooling",
                "token_gate",
                "alpha_param",
                "projector",
                "norm"
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
        self.consistency_lambda = getattr(opt, "consistency_lambda", 0.0)
        self.consistency_warmup = getattr(opt, "consistency_warmup", 0.0)
        self.consistency_noise_std = getattr(opt, "consistency_noise_std", 0.0)
        self.consistency_blur_prob = getattr(opt, "consistency_blur_prob", 0.0)
        self.consistency_blur_sigma_min = getattr(opt, "consistency_blur_sigma_min", 0.0)
        self.consistency_blur_sigma_max = getattr(opt, "consistency_blur_sigma_max", 0.0)
        self.consistency_resize_scale = getattr(opt, "consistency_resize_scale", 0.0)
        self.consistency_total_steps = None


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

    def set_consistency_total_steps(self, total_steps):
        self.consistency_total_steps = max(int(total_steps), 0)

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def _get_module(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def _consistency_weight(self):
        if self.consistency_lambda <= 0:
            return 0.0
        if not self.consistency_total_steps or self.consistency_warmup <= 0:
            return self.consistency_lambda
        warmup_steps = int(self.consistency_total_steps * self.consistency_warmup)
        warmup_steps = max(warmup_steps, 1)
        scale = min(1.0, float(self.total_steps) / warmup_steps)
        return self.consistency_lambda * scale

    def _gaussian_blur(self, x, sigma):
        if sigma <= 0:
            return x
        if sigma < 0.8:
            ksize = 3
        else:
            ksize = 5
        half = ksize // 2
        coords = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, ksize, ksize)
        kernel = kernel_2d.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=half, groups=x.shape[1])

    def _consistency_augment(self, x):
        out = x
        if self.consistency_resize_scale > 0:
            scale = 1.0 + (torch.rand(1, device=x.device).item() * 2 - 1) * self.consistency_resize_scale
            h, w = out.shape[-2:]
            nh = max(1, int(round(h * scale)))
            nw = max(1, int(round(w * scale)))
            out = F.interpolate(out, size=(nh, nw), mode="bilinear", align_corners=False)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        if self.consistency_blur_prob > 0 and torch.rand(1, device=x.device).item() < self.consistency_blur_prob:
            sigma_min = max(self.consistency_blur_sigma_min, 0.0)
            sigma_max = max(self.consistency_blur_sigma_max, sigma_min)
            sigma = float(torch.empty(1, device=x.device).uniform_(sigma_min, sigma_max).item())
            out = self._gaussian_blur(out, sigma)

        if self.consistency_noise_std > 0:
            noise_std = float(torch.empty(1, device=x.device).uniform_(0.0, self.consistency_noise_std).item())
            out = out + torch.randn_like(out) * noise_std
        return out
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
            if self.consistency_lambda > 0:
                module = self._get_module()
                if hasattr(module, "denoise_and_normalize") and hasattr(module, "forward_denoised"):
                    xw = module.denoise_and_normalize(self.input)
                    xw1 = self._consistency_augment(xw)
                    xw2 = self._consistency_augment(xw)
                    pred1, _, proj1 = module.forward_denoised(xw1, return_projected=True)
                    with torch.no_grad():
                        _, _, proj2 = module.forward_denoised(xw2, return_projected=True)
                    loss_bce = self.loss_fn(pred1.squeeze(1), self.label)
                    z1 = F.normalize(proj1, dim=1)
                    z2 = F.normalize(proj2, dim=1)
                    cons_loss = 1.0 - (z1 * z2).sum(dim=1).mean()
                    weight = self._consistency_weight()
                    return loss_bce + weight * cons_loss
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
        if self.opt.arch.startswith("RFNT") and self.consistency_lambda > 0:
            self.loss = self.get_loss()
        else:
            self.forward()
            self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
