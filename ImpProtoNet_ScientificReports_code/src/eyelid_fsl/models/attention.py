"""Residual channel and spatial attention before global average pooling."""

from __future__ import annotations

import torch

from torch import nn

from torchvision.models import ResNet18_Weights, resnet18

class ResidualCBAM(nn.Module):
    """Channel and spatial attention with an exactly neutral initialization."""

    def __init__(
        self,
        channels: int,
        *,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction_ratio)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.spatial_projection = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2, bias=True,
        )
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)
        nn.init.zeros_(self.spatial_projection.weight)
        nn.init.zeros_(self.spatial_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        avg_desc = features.mean(dim=(2, 3), keepdim=True)
        max_desc = features.amax(dim=(2, 3), keepdim=True)
        channel_logits = self.channel_mlp(avg_desc) + self.channel_mlp(max_desc)
        attended = features * (0.5 + torch.sigmoid(channel_logits))
        spatial_desc = torch.cat(
            (attended.mean(dim=1, keepdim=True), attended.amax(dim=1, keepdim=True)),
            dim=1,
        )
        spatial_logits = self.spatial_projection(spatial_desc)
        return attended * (0.5 + torch.sigmoid(spatial_logits))

def build_attention(name: str, channels: int, **kwargs) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name != "residual_cbam":
        raise ValueError(f"Unsupported attention type: {name}")
    return ResidualCBAM(channels, **kwargs)


class AttentionResNet18Encoder(nn.Module):
    """ResNet18 with configurable residual attention before global average pooling."""

    output_channels = 512
    output_dimension = 512

    def __init__(
        self,
        *,
        pretrained: bool,
        frozen: bool = False,
        attention_type: str = "residual_cbam",
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        attention_kwargs = {}
        if attention_type == "residual_cbam":
            attention_kwargs["reduction_ratio"] = reduction_ratio
        if attention_type == "residual_cbam":
            attention_kwargs["spatial_kernel_size"] = spatial_kernel_size
        
        self.attention = build_attention(attention_type, self.output_channels, **attention_kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pretrained = pretrained
        self.frozen = frozen
        self.attention_type = attention_type
        self.reduction_ratio = int(reduction_ratio)
        self.spatial_kernel_size = int(spatial_kernel_size)
        if frozen:
            self.requires_grad_(False)

    def train(self, mode: bool = True) -> "AttentionResNet18Encoder":
        super().train(mode)
        if self.frozen:
            self.eval()
        return self

    def backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        backbone = self.backbone
        features = backbone.conv1(images)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        features = backbone.layer4(features)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.attention(self.backbone_features(images))
        features = torch.flatten(self.pool(features), start_dim=1)
        return features

def build_attention_resnet18_encoder(
    *,
    pretrained: bool = True,
    frozen: bool = False,
    attention_type: str = "residual_cbam",
    reduction_ratio: int = 16,
    spatial_kernel_size: int = 7,
) -> AttentionResNet18Encoder:
    return AttentionResNet18Encoder(
        pretrained=pretrained, frozen=frozen,
        attention_type=attention_type,
        reduction_ratio=reduction_ratio,
        spatial_kernel_size=spatial_kernel_size,
    )
