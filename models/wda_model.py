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
# NPR-style Residual Attention (legacy, currently unused)
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
    def __init__(self, block=BasicBlock, layers=(2, 2), alpha=1.0, smooth=True,
                 use_abs=True, scale_residual=True):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.use_abs = use_abs
        self.scale_residual = scale_residual

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.att_head = nn.Conv2d(128 * block.expansion, 1, kernel_size=1, bias=True)

        if self.smooth:
            self.smoother = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.constant_(self.smoother.weight, 1.0 / 9.0)

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
        _, C, H, W = r.shape
        assert C == 3

        if self.use_abs:
            r = r.abs()
        if self.scale_residual:
            r = r * (2.0 / 3.0)

        x = self.relu(self.bn1(self.conv1(r)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        att_logits = self.att_head(x)
        A = torch.sigmoid(att_logits)
        A = F.interpolate(A, size=(H, W), mode="bilinear", align_corners=False)

        if self.smooth:
            A = self.smoother(A).clamp(0.0, 1.0)
        return A


# -----------------------------
# Residual -> Token Gate
# -----------------------------
class ResidualTokenGate(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=16, smooth=True, smooth_trainable=False, temperature=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temperature = float(temperature)
        if smooth:
            self.smoother = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            with torch.no_grad():
                self.smoother.weight.fill_(1.0 / 9.0)
            if not smooth_trainable:
                for p in self.smoother.parameters():
                    p.requires_grad_(False)
        else:
            self.smoother = None

    def forward(self, p: torch.Tensor, grid_size):
        logits = self.mlp(p)  # [B,N,1]
        if self.smoother is not None:
            B, N, _ = logits.shape
            hp, wp = grid_size
            logits_2d = logits.transpose(1, 2).reshape(B, 1, hp, wp)
            logits_2d = self.smoother(logits_2d)
            logits = logits_2d.flatten(2).transpose(1, 2)

        temp = max(self.temperature, 1e-6)
        w = torch.softmax(logits / temp, dim=1)
        w = w * w.shape[1]
        return w


class WDAModel(nn.Module):
    def __init__(self,
                 backbone_name,
                 num_classes=1,
                 alpha=1.0,
                 alpha_max=10.0,
                 gate_hidden_dim=16,
                 gate_smooth=True,
                 gate_smooth_trainable=False,
                 residual_embed_dim=4,
                 residual_tau=0.15,
                 eps=1e-6,
                 use_weighted_pool=False,
                 norm_type="bn"):
        super(WDAModel, self).__init__()
        self.bk_name = backbone_name
        self.residual_tau = float(residual_tau)
        self.residual_embed_dim = int(residual_embed_dim)
        self.alpha_max = alpha_max
        self.eps = float(eps)
        self.use_weighted_pool = use_weighted_pool

        if backbone_name.startswith("CLIP"):
            backbone_name_ = backbone_name[5:]
            self.pre_model, self.preprocess = clip.load(backbone_name_, device="cpu")
            feat_dim = CHANNELS[backbone_name_]

            self.use_token_gate = (
                hasattr(self.pre_model, "encode_image_tokens")
                and hasattr(self.pre_model, "visual")
                and hasattr(self.pre_model.visual, "transformer")
            )

            if self.use_token_gate and hasattr(self.pre_model.visual, "conv1") and hasattr(self.pre_model.visual.conv1, "stride"):
                try:
                    self.patch_size = int(self.pre_model.visual.conv1.stride[0])
                except Exception:
                    self.patch_size = None
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

        if norm_type == "bn":
            self.bn = nn.BatchNorm1d(feat_dim)
        elif norm_type == "ln":
            self.bn = nn.LayerNorm(feat_dim)
        elif norm_type in ("none", "identity"):
            self.bn = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.ClassifyNet = nn.Sequential(nn.Linear(feat_dim, num_classes))

        if self.residual_embed_dim > 0:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(3, self.residual_embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GELU(),
            )
        else:
            self.residual_conv = None

        self.token_gate = ResidualTokenGate(
            in_dim=max(1, self.residual_embed_dim),
            hidden_dim=gate_hidden_dim,
            smooth=gate_smooth,
            smooth_trainable=gate_smooth_trainable
        )

        a = float(alpha)
        if a <= 0:
            alpha_init = -10.0
        else:
            alpha_init = math.log(math.expm1(a) + 1e-12)
        self.alpha_param = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def _denormalize(self, x):
        return x * self.pixel_std + self.pixel_mean

    def _normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def _get_alpha(self):
        alpha = F.softplus(self.alpha_param)
        if self.alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(self.alpha_max))
        return alpha

    def backbone_extract(self, x):
        if self.bk_name.startswith("DINOv2"):
            out = self.pre_model(x)
            return out
        return self.pre_model.encode_image(x)

    def _residual_to_tokens(self, r_raw, grid_size):
        tau = max(self.residual_tau, 1e-8)
        r_norm = torch.tanh(r_raw / tau)

        if self.residual_conv is not None:
            feat = self.residual_conv(r_norm)  # [B,Cp,H,W]
        else:
            feat = torch.linalg.vector_norm(r_norm, ord=2, dim=1, keepdim=True)  # [B,1,H,W]

        if self.patch_size is not None:
            pooled = F.avg_pool2d(feat, kernel_size=self.patch_size, stride=self.patch_size)
            if pooled.shape[-2:] != grid_size:
                pooled = F.adaptive_avg_pool2d(feat, grid_size)
        else:
            pooled = F.adaptive_avg_pool2d(feat, grid_size)

        p = pooled.flatten(2).transpose(1, 2)  # [B,N,Cp]
        return p

    def denoise(self, image):
        is_torch = isinstance(image, torch.Tensor)
        if is_torch:
            device = image.device
            dtype = image.dtype
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image

        if len(image_np.shape) == 4:
            denoised_batch = []
            for b in range(image_np.shape[0]):
                denoised_batch.append(self._denoise_single(image_np[b]))
            result = np.stack(denoised_batch, axis=0)
        elif len(image_np.shape) == 3:
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

        return np.stack(denoised_channels, axis=0)

    def forward(self, x, return_feature=False, return_attention=False):
        x_raw = self._denormalize(x)
        xw_raw = self.denoise(x_raw)
        r_raw = x_raw - xw_raw
        xw = self._normalize(xw_raw)

        if self.use_token_gate:
            tokens = self.pre_model.encode_image_tokens(xw)
            patch_tokens = tokens[:, 1:, :]

            _, num_tokens, _ = patch_tokens.shape
            hp = int(math.sqrt(num_tokens))
            wp = num_tokens // hp
            if hp * wp != num_tokens:
                raise ValueError(f"Token grid is not square: N={num_tokens}")

            p = self._residual_to_tokens(r_raw, grid_size=(hp, wp))
            w = self.token_gate(p, grid_size=(hp, wp))
            alpha = self._get_alpha()

            gated_tokens = patch_tokens * (1.0 + alpha * w)
            if self.use_weighted_pool:
                w_sum = w.sum(dim=1, keepdim=True).clamp_min(self.eps)
                z = (gated_tokens * w).sum(dim=1) / w_sum.squeeze(1)
            else:
                z = gated_tokens.mean(dim=1)

            feat_bn = self.bn(z)
            pred = self.ClassifyNet(feat_bn)
            return pred, z

        feat = self.backbone_extract(xw)
        feat_bn = self.bn(feat)
        pred = self.ClassifyNet(feat_bn)
        return pred, feat_bn

