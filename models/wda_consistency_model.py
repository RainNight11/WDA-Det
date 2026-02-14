from .clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from .wavelet_utils import validate_wavelet_name


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
    def __init__(self, backbone_name, num_classes=1, proj_dim=256, norm_type="bn",
                 wavelet_name="db4", wavelet_levels=3, wavelet_theta_init=0.02,
                 learn_wavelet=False):
        super(WDAModel, self).__init__()
        self.bk_name = backbone_name
        self.wavelet_name = validate_wavelet_name(wavelet_name)
        self.wavelet_levels = int(wavelet_levels)
        self.learn_wavelet = bool(learn_wavelet)

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

        # ---- Learnable wavelet shrinkage (Scheme A) ----
        wavelet = pywt.Wavelet(self.wavelet_name)
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(wavelet.rec_lo[::-1], dtype=torch.float32)
        rec_hi = torch.tensor(wavelet.rec_hi[::-1], dtype=torch.float32)

        ll = torch.ger(dec_lo, dec_lo)
        lh = torch.ger(dec_lo, dec_hi)
        hl = torch.ger(dec_hi, dec_lo)
        hh = torch.ger(dec_hi, dec_hi)
        dwt_filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,k,k]

        rll = torch.ger(rec_lo, rec_lo)
        rlh = torch.ger(rec_lo, rec_hi)
        rhl = torch.ger(rec_hi, rec_lo)
        rhh = torch.ger(rec_hi, rec_hi)
        iwt_filters = torch.stack([rll, rlh, rhl, rhh], dim=0).unsqueeze(1)

        self.register_buffer("dwt_filters", dwt_filters)
        self.register_buffer("iwt_filters", iwt_filters)
        self.wavelet_pad = int(dec_lo.numel() // 2 - 1)

        theta_init = torch.full((self.wavelet_levels, 3, 3), float(wavelet_theta_init))
        self.wavelet_theta = nn.Parameter(theta_init)
        self.wavelet_theta.requires_grad_(self.learn_wavelet)

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

    # -------- Wavelet denoise (learnable soft-threshold) --------
    def _soft_threshold(self, x, theta):
        return torch.sign(x) * F.relu(torch.abs(x) - theta)

    def _dwt2(self, x):
        B, C, H, W = x.shape
        pad = self.wavelet_pad
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        filters = self.dwt_filters.to(device=x.device, dtype=x.dtype)
        weight = filters.repeat(C, 1, 1, 1)  # [4*C,1,k,k]
        y = F.conv2d(x, weight, stride=2, padding=0, groups=C)
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, (lh, hl, hh), (H, W)

    def _idwt2(self, ll, details, out_size):
        lh, hl, hh = details
        B, C, H, W = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, 4 * C, H, W)
        filters = self.iwt_filters.to(device=ll.device, dtype=ll.dtype)
        weight = filters.repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(y, weight, stride=2, padding=0, groups=C)
        pad = self.wavelet_pad
        if pad > 0:
            h, w = out_size
            x = x[..., pad:pad + h, pad:pad + w]
        return x

    def _denoise_pywt(self, image: torch.Tensor):
        device = image.device
        dtype = image.dtype
        image_np = image.detach().cpu().numpy()

        if image_np.ndim == 4:
            denoised_batch = []
            for b in range(image_np.shape[0]):
                denoised_batch.append(self._denoise_single_pywt(image_np[b]))
            result = np.stack(denoised_batch, axis=0)
        elif image_np.ndim == 3:
            result = self._denoise_single_pywt(image_np)
        else:
            raise ValueError(f"Unsupported input shape: {image_np.shape}")

        return torch.from_numpy(result).to(device=device, dtype=dtype)

    def _denoise_single_pywt(self, image_np):
        denoised_channels = []
        C, H, W = image_np.shape
        for i in range(C):
            coeffs = pywt.wavedec2(image_np[i], self.wavelet_name, level=self.wavelet_levels)
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

            denoised = pywt.waverec2(coeffs_denoised, self.wavelet_name)
            denoised_channels.append(denoised[:H, :W])

        return np.stack(denoised_channels, axis=0)

    def denoise(self, image):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Wavelet denoise expects torch.Tensor input.")
        if image.dim() != 4:
            raise ValueError(f"Unsupported input shape: {image.shape}")
        if image.shape[1] != 3:
            raise ValueError("Wavelet denoise currently supports 3-channel images only.")

        if not self.learn_wavelet:
            return self._denoise_pywt(image)

        ll = image
        details_list = []
        sizes = []

        for level in range(self.wavelet_levels):
            ll, (lh, hl, hh), size = self._dwt2(ll)
            sizes.append(size)
            theta = F.softplus(self.wavelet_theta[level])  # [3,3]
            theta_lh = theta[0].view(1, 3, 1, 1)
            theta_hl = theta[1].view(1, 3, 1, 1)
            theta_hh = theta[2].view(1, 3, 1, 1)
            lh = self._soft_threshold(lh, theta_lh)
            hl = self._soft_threshold(hl, theta_hl)
            hh = self._soft_threshold(hh, theta_hh)
            details_list.append((lh, hl, hh))

        for level in reversed(range(self.wavelet_levels)):
            ll = self._idwt2(ll, details_list[level], sizes[level])
        return ll


    # -------- Forward (single-path inference) --------
    def forward(self, x, return_feature=False, return_attention=False):
        xw = self.denoise_and_normalize(x)
        pred, feat = self.forward_denoised(xw, return_projected=False)
        return pred, feat
