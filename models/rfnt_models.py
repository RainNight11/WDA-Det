from .clip import clip
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    'DINOv2:ViT-S14': 384,
    'DINOv2:ViT-B14': 768,
    'DINOv2:ViT-L14': 1024,
    'DINOv2:ViT-G14': 1536,
}

PIXEL_MEAN = {
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "dinov2": [0.5, 0.5, 0.5],
    "imagenet": [0.485, 0.456, 0.406],
}

PIXEL_STD = {
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "dinov2": [0.5, 0.5, 0.5],
    "imagenet": [0.229, 0.224, 0.225],
}

# -----------------------------
# NPR-style Residual Attention
# -----------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class NPRResidualAttention(nn.Module):
    """
    用 NPR 风格的 ResNet 前半段（conv1/maxpool/layer1/layer2）
    把 residual r = x - x_tilde 编码成空间注意力图 A。

    输入:
        r: [B,3,H,W] residual (可为 signed 或 abs)
    输出:
        A: [B,1,H,W] in [0,1]
    """
    def __init__(self, block=BasicBlock, layers=(2, 2), alpha=1.0, smooth=True,
                 use_abs=True, scale_residual=True):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.use_abs = use_abs
        self.scale_residual = scale_residual  # 是否复用 NPR 论文实现里的 (2/3) 量级

        self.inplanes = 64
        # 对齐你给的 NPR-ResNet：3x3 s2 + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 用 1x1 将 feature map -> attention logits（低分辨率）
        self.att_head = nn.Conv2d(128 * block.expansion, 1, kernel_size=1, bias=True)

        if self.smooth:
            self.smoother = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.constant_(self.smoother.weight, 1.0 / 9.0)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, r: torch.Tensor):
        B, C, H, W = r.shape
        assert C == 3

        if self.use_abs:
            r = r.abs()

        # 可选：对齐你之前 NPR 输入缩放 (NPR*2/3)
        if self.scale_residual:
            r = r * (2.0 / 3.0)

        x = self.relu(self.bn1(self.conv1(r)))   # [B,64,H/2,W/2]
        x = self.maxpool(x)                      # [B,64,H/4,W/4]
        x = self.layer1(x)                       # [B,64,H/4,W/4]
        x = self.layer2(x)                       # [B,128,H/8,W/8]

        att_logits = self.att_head(x)            # [B,1,H/8,W/8]
        A = torch.sigmoid(att_logits)            # [B,1,H/8,W/8]
        A = F.interpolate(A, size=(H, W), mode="bilinear", align_corners=False)

        if self.smooth:
            A = self.smoother(A).clamp(0.0, 1.0)

        return A


class ResidualTokenGate(nn.Module):
    def __init__(self, hidden_dim=16, smooth=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if smooth:
            self.smoother = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.constant_(self.smoother.weight, 1.0 / 9.0)
        else:
            self.smoother = None

    def forward(self, p: torch.Tensor, grid_size):
        """
        p: [B, N, 1]
        grid_size: (Hp, Wp)
        """
        w = self.mlp(p)
        if self.smoother is not None:
            B, N, _ = w.shape
            hp, wp = grid_size
            w_2d = w.transpose(1, 2).reshape(B, 1, hp, wp)
            w_2d = self.smoother(w_2d)
            w = w_2d.flatten(2).transpose(1, 2)
        return w


class RFNTModel(nn.Module):
    """
    Token-wise feature gating version:
        x -> wavelet denoise -> x_w (clean)
        r = x_raw - x_w_raw
        E = backbone_tokens(x_w)  # patch tokens only
        W = gate(r)               # token-wise gate
        E' = E * (1 + alpha * W)
        z = mean(E')
        z -> BN -> Linear -> pred
    """

    def __init__(self, backbone_name, num_classes=1,
                 alpha=1.0,
                 gate_hidden_dim=16,
                 gate_smooth=True,
                 residual_tau=0.15):
        super(RFNTModel, self).__init__()
        self.bk_name = backbone_name
        self.residual_tau = residual_tau

        # backbone
        if backbone_name.startswith("CLIP"):
            backbone_name = backbone_name[5:]  # e.g. "ViT-L/14"
            self.pre_model, self.preprocess = clip.load(backbone_name, device="cpu")
            feat_dim = CHANNELS[backbone_name]
            self.use_token_gate = (
                hasattr(self.pre_model, "encode_image_tokens")
                and hasattr(self.pre_model, "visual")
                and hasattr(self.pre_model.visual, "transformer")
            )
            if self.use_token_gate and hasattr(self.pre_model.visual, "conv1"):
                self.patch_size = int(self.pre_model.visual.conv1.stride[0])
            else:
                self.patch_size = None
            stat_key = "clip"
        elif backbone_name.startswith("DINOv2"):
            model_tag = backbone_name.split(":")[1]
            model_tag = model_tag.replace("-", "").lower()
            hub_name = f"dinov2_{model_tag}"
            self.pre_model = torch.hub.load('models/dinov2', hub_name, source='local').cuda()
            feat_dim = CHANNELS[backbone_name]
            self.use_token_gate = False
            self.patch_size = None
            stat_key = "dinov2"
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        mean = torch.tensor(PIXEL_MEAN[stat_key]).view(1, 3, 1, 1)
        std = torch.tensor(PIXEL_STD[stat_key]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        self.bn = nn.BatchNorm1d(feat_dim)
        self.ClassifyNet = nn.Sequential(nn.Linear(feat_dim, num_classes))

        self.token_gate = ResidualTokenGate(hidden_dim=gate_hidden_dim, smooth=gate_smooth)
        alpha_init = math.log(math.exp(alpha) - 1.0)
        self.alpha_param = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def backbone_extract(self, x):
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)
        else:
            return self.pre_model.encode_image(x)

    def _denormalize(self, x):
        return x * self.pixel_std + self.pixel_mean

    def _normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def _residual_to_tokens(self, r_raw, grid_size):
        r_norm = torch.tanh(r_raw / self.residual_tau)
        energy = torch.linalg.vector_norm(r_norm, ord=2, dim=1, keepdim=True)
        if self.patch_size is not None:
            pooled = F.avg_pool2d(energy, kernel_size=self.patch_size, stride=self.patch_size)
        else:
            pooled = F.adaptive_avg_pool2d(energy, grid_size)
        if pooled.shape[-2:] != grid_size:
            pooled = F.adaptive_avg_pool2d(energy, grid_size)
        p = pooled.flatten(2).transpose(1, 2)  # [B, N, 1]
        return p

    # -------- Wavelet denoise (db4 + BayesShrink),保持你原实现 --------
    def denoise(self, image):
        is_torch = isinstance(image, torch.Tensor)
        if is_torch:
            device = image.device
            dtype = image.dtype
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image

        if len(image_np.shape) == 4:  # [B,C,H,W]
            denoised_batch = []
            for b in range(image_np.shape[0]):
                denoised_batch.append(self._denoise_single(image_np[b]))
            result = np.stack(denoised_batch, axis=0)
        elif len(image_np.shape) == 3:  # [C,H,W]
            result = self._denoise_single(image_np)
        else:
            raise ValueError(f"Unsupported input shape: {image_np.shape}")

        if is_torch:
            result = torch.from_numpy(result).to(device=device, dtype=dtype)
        return result

    def _denoise_single(self, image):
        denoised_channels = []
        C, H, W = image.shape

        for i in range(C):
            coeffs = pywt.wavedec2(image[i], 'db4', level=3)

            coeffs_denoised = [coeffs[0]]
            for j in range(1, len(coeffs)):
                cH, cV, cD = coeffs[j]
                sigma = np.median(np.abs(cD)) / 0.6745
                sigma_y = np.std(cD)
                sigma_x = np.sqrt(max(sigma_y ** 2 - sigma ** 2, 0))
                threshold = sigma ** 2 / max(sigma_x, 1e-10)

                cH = np.sign(cH) * np.maximum(np.abs(cH) - threshold, 0)
                cV = np.sign(cV) * np.maximum(np.abs(cV) - threshold, 0)
                cD = np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0)

                coeffs_denoised.append((cH, cV, cD))

            denoised = pywt.waverec2(coeffs_denoised, 'db4')
            denoised_channels.append(denoised[:H, :W])

        return np.stack(denoised_channels, axis=0)  # [C,H,W]

    # -------- Forward --------
    def forward(self, x, return_feature=False, return_attention=False):
        """
        x: [B,3,H,W]
        """
        # 1) denoise in pixel space
        x_raw = self._denormalize(x)
        xw_raw = self.denoise(x_raw)  # [B,3,H,W]
        r_raw = x_raw - xw_raw
        xw = self._normalize(xw_raw)

        if self.use_token_gate:
            tokens = self.pre_model.encode_image_tokens(xw)  # [B, 1+N, C]
            patch_tokens = tokens[:, 1:, :]
            bsz, num_tokens, _ = patch_tokens.shape
            hp = int(math.sqrt(num_tokens))
            wp = num_tokens // hp
            if hp * wp != num_tokens:
                raise ValueError(f"Token grid is not square: N={num_tokens}")

            p = self._residual_to_tokens(r_raw, grid_size=(hp, wp))
            w = self.token_gate(p, grid_size=(hp, wp))
            alpha = F.softplus(self.alpha_param)
            gated_tokens = patch_tokens * (1.0 + alpha * w)
            z = gated_tokens.mean(dim=1)
            feat_bn = self.bn(z)
            pred = self.ClassifyNet(feat_bn)
            return pred, z

        feat = self.backbone_extract(xw)  # [B,D]
        feat_bn = self.bn(feat)
        pred = self.ClassifyNet(feat_bn)
        return pred, feat_bn
