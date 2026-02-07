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


class WDAModel(nn.Module):
    """
    Consistency-regularized WDA (single-path inference):
        Inference: x -> denoise -> xw -> f -> g
        Training: consistency between two lightly augmented xw views (stop-grad teacher)
    """
    def __init__(self, backbone_name, num_classes=1, proj_dim=256, norm_type="bn"):
        super(WDAModel, self).__init__()
        self.bk_name = backbone_name

        if backbone_name.startswith("CLIP"):
            backbone_name_ = backbone_name[5:]
            self.pre_model, self.preprocess = clip.load(backbone_name_, device="cpu")
            feat_dim = CHANNELS[backbone_name_]
            stat_key = "clip"
        elif backbone_name.startswith("DINOv2"):
            model_tag = backbone_name.split(":")[1]
            model_tag = model_tag.replace("-", "").lower()
            hub_name = f"dinov2_{model_tag}"
            self.pre_model = torch.hub.load('models/dinov2', hub_name, source='local').cuda()
            feat_dim = CHANNELS[backbone_name]
            stat_key = "dinov2"
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        mean = torch.tensor(PIXEL_MEAN[stat_key]).view(1, 3, 1, 1)
        std = torch.tensor(PIXEL_STD[stat_key]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(feat_dim)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(feat_dim)
        elif norm_type in ("none", "identity"):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.ClassifyNet = nn.Sequential(nn.Linear(feat_dim, num_classes))

        proj_dim = min(int(proj_dim), feat_dim)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    # ---------- small helpers ----------
    def _denormalize(self, x):
        return x * self.pixel_std + self.pixel_mean

    def _normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def denoise_and_normalize(self, x):
        x_raw = self._denormalize(x)
        xw_raw = self.denoise(x_raw)
        return self._normalize(xw_raw)

    def backbone_extract(self, x):
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)
        return self.pre_model.encode_image(x)

    def forward_denoised(self, xw, return_projected=False):
        feat = self.backbone_extract(xw)
        feat_normed = self.norm(feat)
        pred = self.ClassifyNet(feat_normed)
        if return_projected:
            proj = self.projector(feat)
            return pred, feat, proj
        return pred, feat

    # -------- Wavelet denoise (db4 + BayesShrink) --------
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

    # -------- Forward (single-path inference) --------
    def forward(self, x, return_feature=False, return_attention=False):
        xw = self.denoise_and_normalize(x)
        pred, feat = self.forward_denoised(xw, return_projected=False)
        return pred, feat
