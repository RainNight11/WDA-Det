from .clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "DINOv2:ViT-S14": 384,
    "DINOv2:ViT-B14": 768,
    "DINOv2:ViT-L14": 1024,
    "DINOv2:ViT-G14": 1536,
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


class WDADecisionFusionModel(nn.Module):
    """
    Main + auxiliary branch with decision-level fusion.

    Main branch:
        x -> denoise -> xw -> backbone -> main_head -> s_main

    Auxiliary branch:
        r = x_raw - xw_raw -> aux_cnn -> aux_head -> s_aux
                              aux_gate -> q in [0,1]

    Final decision:
        s_final = s_main + gamma * q * tanh(s_aux)

    Notes:
    - Residual information is NOT injected back into the input of backbone.
    - gamma is initialized at 0 for stable start.
    - This file is standalone and not wired into models/__init__.py by default.
    """

    def __init__(
        self,
        backbone_name,
        num_classes=1,
        norm_type="bn",
        wavelet_name="db4",
        wavelet_levels=3,
        wavelet_theta_init=0.02,
        learn_wavelet=False,
        aux_channels=128,
    ):
        super(WDADecisionFusionModel, self).__init__()
        self.bk_name = backbone_name
        self.wavelet_name = wavelet_name
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
            self.pre_model = torch.hub.load("models/dinov2", hub_name, source="local").cuda()
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

        # Main branch heads
        self.main_head = nn.Linear(feat_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, min(256, feat_dim)),
            nn.GELU(),
            nn.Linear(min(256, feat_dim), min(256, feat_dim)),
        )

        # Auxiliary branch on residual cue
        self.aux_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, aux_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(aux_channels),
            nn.ReLU(inplace=True),
        )
        self.aux_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.aux_head = nn.Linear(aux_channels, num_classes)
        self.aux_gate = nn.Sequential(
            nn.Linear(aux_channels + feat_dim, aux_channels),
            nn.GELU(),
            nn.Linear(aux_channels, 1),
            nn.Sigmoid(),
        )

        # Fusion strength (starts at 0 => behaves as pure main branch initially)
        self.fusion_gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # ---- Learnable wavelet branch buffers/params (optional) ----
        wavelet = pywt.Wavelet(self.wavelet_name)
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(wavelet.rec_lo[::-1], dtype=torch.float32)
        rec_hi = torch.tensor(wavelet.rec_hi[::-1], dtype=torch.float32)

        ll = torch.ger(dec_lo, dec_lo)
        lh = torch.ger(dec_lo, dec_hi)
        hl = torch.ger(dec_hi, dec_lo)
        hh = torch.ger(dec_hi, dec_hi)
        dwt_filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)

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

    def _denormalize(self, x):
        return x * self.pixel_std + self.pixel_mean

    def _normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def backbone_extract(self, x):
        if self.bk_name.startswith("DINOv2"):
            return self.pre_model(x)
        return self.pre_model.encode_image(x)

    def _soft_threshold(self, x, theta):
        return torch.sign(x) * F.relu(torch.abs(x) - theta)

    def _dwt2(self, x):
        bsz, channels, h, w = x.shape
        pad = self.wavelet_pad
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        filters = self.dwt_filters.to(device=x.device, dtype=x.dtype)
        weight = filters.repeat(channels, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=channels)
        y = y.view(bsz, channels, 4, y.shape[-2], y.shape[-1])
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, (lh, hl, hh), (h, w)

    def _idwt2(self, ll, details, out_size):
        lh, hl, hh = details
        bsz, channels, h, w = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).reshape(bsz, 4 * channels, h, w)
        filters = self.iwt_filters.to(device=ll.device, dtype=ll.dtype)
        weight = filters.repeat(channels, 1, 1, 1)
        x = F.conv_transpose2d(y, weight, stride=2, padding=0, groups=channels)
        pad = self.wavelet_pad
        if pad > 0:
            oh, ow = out_size
            x = x[..., pad:pad + oh, pad:pad + ow]
        return x

    def _denoise_pywt(self, image):
        device = image.device
        dtype = image.dtype
        image_np = image.detach().cpu().numpy()

        if image_np.ndim == 4:
            denoised_batch = []
            for bsz in range(image_np.shape[0]):
                denoised_batch.append(self._denoise_single_pywt(image_np[bsz]))
            result = np.stack(denoised_batch, axis=0)
        elif image_np.ndim == 3:
            result = self._denoise_single_pywt(image_np)
        else:
            raise ValueError(f"Unsupported input shape: {image_np.shape}")

        return torch.from_numpy(result).to(device=device, dtype=dtype)

    def _denoise_single_pywt(self, image_np):
        denoised_channels = []
        channels, h, w = image_np.shape
        for i in range(channels):
            coeffs = pywt.wavedec2(image_np[i], self.wavelet_name, level=self.wavelet_levels)
            coeffs_denoised = [coeffs[0]]
            for j in range(1, len(coeffs)):
                c_h, c_v, c_d = coeffs[j]
                sigma = np.median(np.abs(c_d)) / 0.6745
                sigma_y = np.std(c_d)
                sigma_x = np.sqrt(max(sigma_y ** 2 - sigma ** 2, 0))
                threshold = sigma ** 2 / max(sigma_x, 1e-10)

                c_h = np.sign(c_h) * np.maximum(np.abs(c_h) - threshold, 0)
                c_v = np.sign(c_v) * np.maximum(np.abs(c_v) - threshold, 0)
                c_d = np.sign(c_d) * np.maximum(np.abs(c_d) - threshold, 0)
                coeffs_denoised.append((c_h, c_v, c_d))

            denoised = pywt.waverec2(coeffs_denoised, self.wavelet_name)
            denoised_channels.append(denoised[:h, :w])

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

    def denoise_and_normalize(self, x):
        x_raw = self._denormalize(x)
        xw_raw = self.denoise(x_raw)
        xw = self._normalize(xw_raw)
        return xw, x_raw, xw_raw

    def forward(self, x, return_feature=False, return_attention=False):
        xw, x_raw, xw_raw = self.denoise_and_normalize(x)
        residual = x_raw - xw_raw

        # Main branch
        feat_main = self.backbone_extract(xw)
        feat_main_norm = self.norm(feat_main)
        s_main = self.main_head(feat_main_norm)

        # Auxiliary branch
        aux_feat_map = self.aux_branch(residual)
        aux_feat = self.aux_pool(aux_feat_map).flatten(1)
        s_aux = self.aux_head(aux_feat)

        # Confidence gate conditioned on both branches
        gate_input = torch.cat([aux_feat, feat_main_norm], dim=1)
        q = self.aux_gate(gate_input)  # [B,1]

        # Decision-level fusion (no residual reinjection to main input)
        s_final = s_main + self.fusion_gamma * q * torch.tanh(s_aux)

        if return_attention:
            attention = aux_feat_map.mean(dim=1, keepdim=True)
            attention = F.interpolate(attention, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if return_feature:
                return s_final, feat_main_norm, {
                    "main_logit": s_main,
                    "aux_logit": s_aux,
                    "gate": q,
                    "attention": attention,
                }
            return s_final, attention

        if return_feature:
            return s_final, feat_main_norm, {
                "main_logit": s_main,
                "aux_logit": s_aux,
                "gate": q,
            }

        return s_final, feat_main_norm
