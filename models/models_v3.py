import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import clip


# --- Re-usable Feature Metric ---

class FeatureMetric(nn.Module):
    """
    A wrapper for a frozen backbone model (like CLIP) to extract features.
    """

    def __init__(self, model_name):
        super(FeatureMetric, self).__init__()
        if not model_name.startswith('CLIP:'):
            raise ValueError("FeatureMetric only supports CLIP models for now.")

        clip_model_name = model_name.split(':')[1]
        self.model, _ = clip.load(clip_model_name, device="cpu")
        self.model = self.model.visual

        # Freeze the entire model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.model.eval()
        return self.model(x)


# --- PGD-based Noise Generator ---

class PgdNoiseGenerator(nn.Module):
    """
    A 'generator' that applies a PGD attack to images.
    This module does not have trainable parameters. Its purpose is to apply a
    procedural attack based on the configuration provided during runtime.
    """

    def __init__(self, feature_model, eps, alpha, steps):
        super(PgdNoiseGenerator, self).__init__()
        self.feature_model = feature_model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

        # This module has no trainable parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        Performs a PGD attack on a batch of images to maximize the feature space distance.
        The attack logic is adapted from validate_pgd.py.
        """
        device = images.device
        original_images = images.detach().clone()

        # Get original features (do not compute gradients for these)
        with torch.no_grad():
            original_features = self.feature_model(original_images).detach()

        # Initialize adversarial images with a small random perturbation
        adv_images = images.detach().clone() + torch.zeros_like(images).uniform_(-self.eps, self.eps)

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Extract features from the current adversarial images
            adv_features = self.feature_model(adv_images)

            # The loss is the negative cosine similarity. Minimizing this maximizes the distance.
            loss = -F.cosine_similarity(adv_features, original_features).sum()

            # Calculate gradients
            loss.backward()

            # Update images along the gradient sign
            attack_grad = adv_images.grad.detach().sign()
            adv_images = adv_images.detach() + self.alpha * attack_grad

            # Project the perturbation back into the epsilon ball around the original image
            delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
            adv_images = original_images + delta

            # Clamp to a valid normalized image range (e.g., for CLIP)
            adv_images = torch.clamp(adv_images, -2.5, 2.5)

        return adv_images.detach()


# --- Full Detector with PGD-based Perturbation ---

class RFNTDetector_v3(nn.Module):
    def __init__(self, backbone_name, num_classes=2, pgd_params=None):
        super(RFNTDetector_v3, self).__init__()

        # 1. Load the frozen Backbone (e.g., CLIP ViT)
        self.backbone = FeatureMetric(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Initialize Perturbation Generator (PGD Attack)
        if pgd_params:
            self.perturbation_generator = PgdNoiseGenerator(
                feature_model=self.backbone,
                eps=eval(str(pgd_params['eps'])),
                alpha=eval(str(pgd_params['alpha'])),
                steps=int(pgd_params['steps'])
            )
        else:
            # Provide default PGD parameters if not specified
            self.perturbation_generator = PgdNoiseGenerator(
                feature_model=self.backbone,
                eps=8 / 255.0,
                alpha=1 / 255.0,
                steps=10
            )

        # 3. Rebuild the trainable RFNT parts from scratch
        feature_dim = self.backbone.model.output_dim

        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.ClassifyNet = nn.Linear(feature_dim, num_classes)

    def cosine_similarity(self, x1, x2):
        # This logic is copied directly from your original rfnt_models.py
        batch_size = x1.shape[0]
        x1_mean = torch.mean(x1, dim=1, keepdim=True)
        x2_mean = torch.mean(x2, dim=1, keepdim=True)
        x1_centered = x1 - x1_mean
        x2_centered = x2 - x2_mean
        cov_matrices = []
        for i in range(batch_size):
            cov_matrix = torch.mm(x1_centered[i].unsqueeze(1), x2_centered[i].unsqueeze(0)) / (x1.shape[1] - 1)
            cov_matrices.append(cov_matrix)
        output = torch.stack(cov_matrices, dim=0)
        return output

    def forward(self, x):
        # In training mode, we must explicitly set the frozen parts to eval()
        if self.training:
            self.perturbation_generator.eval()
            self.backbone.eval()

        # 1. Use the PGD generator to transform the image.
        #    注意：PGD 攻击内部需要梯度来优化扰动，所以不能使用 torch.no_grad()
        x_transformed = self.perturbation_generator(x)

        # 2. 提取原图和扰动图的特征（不需要梯度）
        with torch.no_grad():
            feat_orig = self.backbone(x)
            feat_transformed = self.backbone(x_transformed)

        # 3. Calculate feature similarity (the trainable part)
        # Gradient calculation starts from here.
        cos_sim = self.cosine_similarity(feat_orig, feat_transformed)

        # 4. Classification head
        out = self.pooling(cos_sim)
        out = self.bn(out.squeeze(-1))
        pred = self.ClassifyNet(out)
        return pred
