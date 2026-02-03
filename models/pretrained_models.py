import torch
import torch.nn as nn
from .clip import clip
from .rfnt_models import RFNTModel  # We will reuse parts of your original model


# --- Model for Stage 1 ---

# Using your original NoiseGenerator, but renamed for consistency.
# The final activation is changed from ReLU to Tanh to allow for both positive and negative perturbations.
class PerturbationGenerator(nn.Module):
    def __init__(self, num_channels=3, noise_channels=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 32, 3, padding=1, dilation=2)

        self.trans1 = torch.nn.ConvTranspose2d(32, 128, 3, padding=1, dilation=2)
        self.trans2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(32, 3, 3, padding=1)
        self.mp = torch.nn.MaxPool2d(2, return_indices=True)
        self.up = torch.nn.MaxUnpool2d(2)
        self.relu = torch.nn.ReLU()

    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # [?, 32, 224, 224]
        s1 = x.size()
        x, ind1 = self.mp(x)  # [?, 32, 112, 112]
        x = self.conv2(x)
        x = self.relu(x)  # [?, 64, 112, 112]
        s2 = x.size()
        x, ind2 = self.mp(x)  # [?, 64, 56, 56]
        x = self.conv3(x)
        x = self.relu(x)  # [?, 128, 56, 56]
        s3 = x.size()
        x, ind3 = self.mp(x)  # [?, 128, 28, 28]
        x = self.conv4(x)
        x = self.relu(x)  # [?, 32, 28, 28]

        return x, ind1, s1, ind2, s2, ind3, s3

    def decoder(self, x, ind1, s1, ind2, s2, ind3, s3):
        x = self.trans1(x)
        x = self.relu(x)  # [128, 128, 28, 28]
        x = self.up(x, ind3, output_size=s3)  # [128, 128, 56, 56]
        x = self.trans2(x)
        x = self.relu(x)  # [128, 128, 56, 56]
        x = self.up(x, ind2, output_size=s2)  # [128, 128, 112, 112]
        x = self.trans3(x)
        x = self.relu(x)  # [128, 128, 112, 112]
        x = self.up(x, ind1, output_size=s1)  # [128, 128, 224, 224]
        x = self.trans4(x)
        x = self.relu(x)  # [128, 128, 224, 224]
        return x

    def forward(self, x):
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        output = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)
        return output


class FeatureMetric(nn.Module):
    """
    A wrapper for a frozen backbone model (like CLIP) to extract features.
    """

    def __init__(self, model_name):
        super(FeatureMetric, self).__init__()
        if model_name.startswith('CLIP:'):
            model_name = model_name.split(':')[1]
            self.model, _ = clip.load(model_name, device="cpu")
            self.model = self.model.visual
        elif model_name.startswith('DINOv2:'):
            model_name = model_name.split(':')[1]
            self.model = torch.hub.load('models/dinov2', model_name,source='local').cuda()
        else:
            raise ValueError("FeatureMetric only supports CLIP models for now.")
        # Freeze the entire model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.model.eval()
        return self.model(x)


# --- Model for Stage 2 ---

class RFNTDetector_v2(nn.Module):
    def __init__(self, backbone_name, num_classes=2, generator_checkpoint_path=None):
        super(RFNTDetector_v2, self).__init__()

        # 1. Initialize Perturbation Generator (load weights if path provided, then freeze)
        self.perturbation_generator = PerturbationGenerator()
        if generator_checkpoint_path:
            # --- FIX: Load the state dict from the checkpoint dictionary ---
            checkpoint = torch.load(generator_checkpoint_path, map_location='cpu')
            self.perturbation_generator.load_state_dict(checkpoint['generator_state_dict'])

        for param in self.perturbation_generator.parameters():
            param.requires_grad = False

        # 2. Load the frozen Backbone (e.g., CLIP ViT)
        self.backbone = FeatureMetric(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3. Rebuild the trainable RFNT parts from scratch
        feature_dim = self.backbone.model.output_dim

        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(feature_dim)
        # FIXED: ClassifyNet now correctly outputs `num_classes` (i.e., 2) features.
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
        # to ensure they behave correctly (e.g., batchnorm, dropout).
        # In eval mode, the top-level model.eval() has already done this.
        if self.training:
            self.perturbation_generator.eval()
            self.backbone.eval()

        # 1. Use the generator to transform the image directly (no gradients needed)
        with torch.no_grad():
            x_transformed = self.perturbation_generator(x)

        # 2. Extract features from original and transformed images (no gradients needed)
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