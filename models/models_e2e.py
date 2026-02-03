# RFNT/models_e2e.py
# Models designed specifically for the End-to-End (e2e) joint training script.
# This version is corrected to be a faithful adaptation of the original RFNTDetector_v2 logic.

import torch
import torch.nn as nn
from .clip import clip


class PerturbationGenerator(nn.Module):
    """
    Generates a perturbation map which is added to the input image.
    The architecture is a U-Net style encoder-decoder.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 32, 3, padding=1, dilation=2)

        self.trans1 = nn.ConvTranspose2d(32, 128, 3, padding=1, dilation=2)
        self.trans2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.trans3 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.trans4 = nn.ConvTranspose2d(32, 3, 3, padding=1)

        self.mp = nn.MaxPool2d(2, return_indices=True)
        self.up = nn.MaxUnpool2d(2)
        self.relu = nn.ReLU()

    def encoder(self, x):
        x = self.relu(self.conv1(x))
        s1 = x.size()
        x, ind1 = self.mp(x)
        x = self.relu(self.conv2(x))
        s2 = x.size()
        x, ind2 = self.mp(x)
        x = self.relu(self.conv3(x))
        s3 = x.size()
        x, ind3 = self.mp(x)
        x = self.relu(self.conv4(x))
        return x, ind1, s1, ind2, s2, ind3, s3

    def decoder(self, x, ind1, s1, ind2, s2, ind3, s3):
        x = self.relu(self.trans1(x))
        x = self.up(x, ind3, output_size=s3)
        x = self.relu(self.trans2(x))
        x = self.up(x, ind2, output_size=s2)
        x = self.relu(self.trans3(x))
        x = self.up(x, ind1, output_size=s1)
        x = self.trans4(x)
        return x

    def forward(self, x):
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        perturbation = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)
        return perturbation


class FeatureMetric(nn.Module):
    """
    A wrapper for a frozen CLIP visual backbone to extract features.
    """

    def __init__(self, model_name, device="cpu"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model = self.model.visual
        self.output_dim = self.model.output_dim

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.model.eval()
        # Get the expected dtype from the model's parameters (e.g., float16 on GPU).
        model_dtype = self.model.conv1.weight.dtype
        # Cast the input tensor to the model's dtype before the forward pass.
        output = self.model(x.to(model_dtype))
        # Always cast the output back to float32 for consistency with the rest of our network.
        return output.to(torch.float32)


def cross_covariance_matrix(x1, x2):
    """
    Calculates the cross-covariance matrix between two batches of feature vectors.
    This is an efficient, batched version of the logic in the original RFNTDetector_v2.
    Input: x1, x2 are tensors of shape (batch_size, feature_dim)
    Output: tensor of shape (batch_size, feature_dim, feature_dim)
    """
    batch_size, feature_dim = x1.shape

    x1_mean = torch.mean(x1, dim=1, keepdim=True)
    x2_mean = torch.mean(x2, dim=1, keepdim=True)
    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    # Reshape for batch matrix multiplication: (B, D, 1) @ (B, 1, D) -> (B, D, D)
    cov_matrices = torch.bmm(x1_centered.unsqueeze(2), x2_centered.unsqueeze(1)) / (feature_dim - 1)

    return cov_matrices


class DetectorHead(nn.Module):
    """
    This is the trainable classification head, faithfully recreating the logic
    from the original RFNTDetector_v2.
    It takes a covariance matrix as input and produces a classification score.
    """

    def __init__(self, feature_dim, num_classes=2):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.classify_net = nn.Linear(feature_dim, num_classes)

    def forward(self, cov_matrix):
        # Input cov_matrix shape: (batch_size, feature_dim, feature_dim)
        # The pooling is applied to the last dimension of the input.
        out = self.pooling(cov_matrix)  # -> (batch_size, feature_dim, 1)
        out = self.bn(out.squeeze(-1))  # -> (batch_size, feature_dim)
        pred = self.classify_net(out)  # -> (batch_size, num_classes)
        return pred
