"""Shared feature backbones used by all comparable experiment models."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Encoder(nn.Module):
    """ResNet18 without its ImageNet classification layer."""

    output_dimension = 512

    def __init__(self, *, pretrained: bool, frozen: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.pretrained = pretrained
        self.frozen = frozen
        if frozen:
            self.requires_grad_(False)

    def train(self, mode: bool = True) -> "ResNet18Encoder":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.output_dimension:
            raise RuntimeError(f"Unexpected ResNet18 feature shape: {features.shape}")
        return features


def build_resnet18_encoder(
    *,
    pretrained: bool = True,
    frozen: bool = False,
) -> ResNet18Encoder:
    """Construct the approved ResNet18 feature extractor configuration."""

    return ResNet18Encoder(pretrained=pretrained, frozen=frozen)


class ResNet18SpatialEncoder(nn.Module):
    """ResNet18 through layer4, preserving the final spatial feature map."""

    output_channels = 512
    output_dimension = 512

    def __init__(self, *, pretrained: bool, frozen: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.pretrained = pretrained
        self.frozen = frozen
        if frozen:
            self.requires_grad_(False)

    def train(self, mode: bool = True) -> "ResNet18SpatialEncoder":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        features = backbone.conv1(images)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        features = backbone.layer4(features)
        if features.ndim != 4 or features.shape[1] != self.output_channels:
            raise RuntimeError(
                f"Unexpected ResNet18 spatial feature shape: {features.shape}"
            )
        return features


def build_resnet18_spatial_encoder(
    *,
    pretrained: bool = True,
    frozen: bool = False,
) -> ResNet18SpatialEncoder:
    """Construct ResNet18 without global pooling for spatial relation learning."""

    return ResNet18SpatialEncoder(pretrained=pretrained, frozen=frozen)
