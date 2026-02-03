from .clip import clip
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


class RFNTModel(nn.Module):
    """
    New Attention version (你要的结构):
        x -> wavelet denoise -> x_tilde (clean)
        r = x - x_tilde
        r -> NPR-style branch -> A
        x_att = x_tilde * (1 + alpha*A)
        x_att -> CLIP/DINOv2 -> feature -> BN -> Linear -> pred
    """

    def __init__(self, backbone_name, num_classes=1,
                 alpha=1.0, smooth=True,
                 npr_layers=(2, 2), npr_use_abs=True, npr_scale_residual=True):
        super(RFNTModel, self).__init__()
        self.bk_name = backbone_name

        # backbone
        if backbone_name.startswith("CLIP"):
            backbone_name = backbone_name[5:]  # e.g. "ViT-L/14"
            self.pre_model, self.preprocess = clip.load(backbone_name, device="cpu")
            feat_dim = CHANNELS[backbone_name]
        elif backbone_name.startswith("DINOv2"):
            model_tag = backbone_name.split(":")[1]
            model_tag = model_tag.replace("-", "").lower()
            hub_name = f"dinov2_{model_tag}"
            self.pre_model = torch.hub.load('models/dinov2', hub_name, source='local').cuda()
            feat_dim = CHANNELS[backbone_name]
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.bn = nn.BatchNorm1d(feat_dim)
        self.ClassifyNet = nn.Sequential(nn.Linear(feat_dim, num_classes))

        # 关键：Residual -> NPR-style attention
        self.res_att = NPRResidualAttention(
            layers=npr_layers,
            alpha=alpha,
            smooth=smooth,
            use_abs=npr_use_abs,
            scale_residual=npr_scale_residual
        )
        self.alpha = alpha  # 给 forward 用

    def backbone_extract(self, x):
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)
        else:
            return self.pre_model.encode_image(x)

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
        # 1) clean image via wavelet denoise
        x_tilde = self.denoise(x)  # [B,3,H,W]

        # # 2) residual map (你要的：x - x_denoised)
        # r = x - x_tilde            # [B,3,H,W]

        # # 3) residual branch -> attention map
        # A = self.res_att(r)        # [B,1,H,W]

        # # 4) attention on clean image
        # x_att = x_tilde * (1.0 + self.alpha * A)

        x_att = x_tilde
        # 5) backbone feature
        feat = self.backbone_extract(x_att)  # [B,D]
        feat_bn = self.bn(feat)

        # if return_feature:
        #     if return_attention:
        #         return feat_bn, A
        #     return feat_bn

        pred = self.ClassifyNet(feat_bn)

        # if return_attention:
        #     return pred, A, x_att
        
        return pred, None

