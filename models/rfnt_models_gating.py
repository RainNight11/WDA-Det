from .clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


CHANNELS = {
    # CLIP:
    "RN50": 1024,
    "ViT-L/14": 768,
    # DINOv2（如果你未来要用，也可保留）
    'DINOv2:ViT-S14': 384,
    'DINOv2:ViT-B14': 768,
    'DINOv2:ViT-L14': 1024,
    'DINOv2:ViT-G14': 1536,
}


class ResidualGuidedImageAttention(nn.Module):
    """
    Lightweight Residual-Guided Image Attention (RGIA)
    在 CLIP 之前对降噪图像 x_tilde 做 residual-guided attention。

    Inputs:
        x:       [B,3,H,W] 原图
        x_tilde: [B,3,H,W] 小波降噪图
    Outputs:
        x_att:   [B,3,H,W] attention-refined denoised image
        A:       [B,1,H,W] 注意力图 (0~1)
    """
    def __init__(self, patch_size: int = 16, alpha: float = 1.0, smooth: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.alpha = alpha
        self.smooth = smooth

        # 轻量 gating： (B,1,Gh,Gw) -> (B,1,Gh,Gw)
        self.gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # 可选：深度可分离平滑（仍然非常轻）
        if self.smooth:
            self.smoother = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.constant_(self.smoother.weight, 1.0 / 9.0)

    def forward(self, x: torch.Tensor, x_tilde: torch.Tensor):
        B, C, H, W = x.shape
        assert C == 3 and x_tilde.shape == x.shape

        # r = |x - x_tilde| : [B,3,H,W]
        r = (x - x_tilde).abs()

        # r_mag = mean over channels : [B,1,H,W]
        r_mag = r.mean(dim=1, keepdim=True)

        # patch pooling to (Gh, Gw)
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "H and W must be divisible by patch_size"
        Gh, Gw = H // P, W // P
        r_grid = F.adaptive_avg_pool2d(r_mag, output_size=(Gh, Gw))  # [B,1,Gh,Gw]

        # gating weights on grid
        W_grid = self.gate(r_grid)  # [B,1,Gh,Gw], in (0,1)

        # upsample to spatial attention A: [B,1,H,W]
        A = F.interpolate(W_grid, size=(H, W), mode="bilinear", align_corners=False)

        # optional smooth refinement
        if self.smooth:
            A = self.smoother(A)
            A = A.clamp(0.0, 1.0)

        # apply attention (broadcast to 3 channels)
        # x_att: [B,3,H,W]
        x_att = x_tilde * (1.0 + self.alpha * A)

        return x_att, A


class RFNTModel(nn.Module):
    """
    Attention version:
        x -> wavelet denoise -> x_tilde
        (x, x_tilde) -> RGIA -> x_att
        x_att -> CLIP -> feature -> BN -> Linear -> pred
    """

    def __init__(self, backbone_name, num_classes=1, patch_size=16, alpha=1.0, smooth=True):
        super(RFNTModel, self).__init__()
        self.bk_name = backbone_name

        # backbone
        if backbone_name.startswith("CLIP"):
            backbone_name = backbone_name[5:]  # e.g. "ViT-L/14"
            self.pre_model, self.preprocess = clip.load(backbone_name, device="cpu")
            feat_dim = CHANNELS[backbone_name]
        elif backbone_name.startswith("DINOv2"):
            # 如果你不需要 DINOv2，可以把这一段删掉
            model_tag = backbone_name.split(":")[1]
            model_tag = model_tag.replace("-", "").lower()
            hub_name = f"dinov2_{model_tag}"
            self.pre_model = torch.hub.load('models/dinov2', hub_name, source='local').cuda()
            feat_dim = CHANNELS[backbone_name]
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.bn = nn.BatchNorm1d(feat_dim)
        self.ClassifyNet = nn.Sequential(
            nn.Linear(feat_dim, num_classes),
        )

        # residual-guided attention module (lightweight)
        self.rgia = ResidualGuidedImageAttention(
            patch_size=patch_size,
            alpha=alpha,
            smooth=smooth
        )

    def backbone_extract(self, x):
        """
        Return global feature: [B, D]
        """
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)  # [B, D]
        else:
            return self.pre_model.encode_image(x)  # [B, D]

    # -------- Wavelet denoise (db4 + BayesShrink),保持你原实现 --------
    def denoise(self, image):
        """
        db4小波降噪函数 - 支持PyTorch GPU张量
        image: [B,C,H,W] or [C,H,W]
        """
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
        """
        image: numpy [C,H,W]
        """
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

                # soft threshold
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
        return_feature: 返回 BN 后特征
        return_attention: 是否返回注意力图 A
        """
        # 1) wavelet denoise
        x_tilde = self.denoise(x)  # [B,3,H,W]

        # 2) residual-guided image attention
        x_att, A = self.rgia(x, x_tilde)  # x_att: [B,3,H,W], A: [B,1,H,W]

        # 3) backbone feature
        feat = self.backbone_extract(x_att)  # [B,D]

        # 4) BN + classifier
        feat_bn = self.bn(feat)  # [B,D]

        if return_feature:
            if return_attention:
                return feat_bn, A
            return feat_bn

        pred = self.ClassifyNet(feat_bn)  # [B,num_classes]

        if return_attention:
            return pred, A
        return pred, None
