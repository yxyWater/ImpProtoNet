"""Shared image encoding, residual adapters and dual-view feature fusion."""
from __future__ import annotations

import torch
from torch import nn

from eyelid_fsl.models.attention import AttentionResNet18Encoder
from eyelid_fsl.models.backbones import ResNet18Encoder, ResNet18SpatialEncoder

class AttentionAdapter(nn.Module):
    """A neutral residual feature adapter after a spatial encoder."""

    def __init__(self, feature_dim: int = 512, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, feature_dim // reduction)
        self.fc1 = nn.Linear(feature_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, feature_dim, bias=False)
        nn.init.zeros_(self.fc2.weight)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(torch.nn.functional.relu(self.fc1(features)))
        return features + self.scale.exp().clamp(max=2.0) * residual

class DualViewAdapter(nn.Module):
    """Zero-initialized residual adapters for a dual-view encoder."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.crop_adapter = AttentionAdapter(feature_dim, reduction=2)
        self.full_adapter = AttentionAdapter(feature_dim, reduction=2)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.crop_adapter(residual)

class DualViewEncoder(nn.Module):
    """Fuse full-frame and soft central-crop representations neutrally."""

    output_dimension = 512

    def __init__(self, base_encoder: nn.Module, mode: str = "central_soft_crop") -> None:
        super().__init__()
        crop_modes = {
            "central_soft_crop": 0.125,
            "central_soft_crop_95": 0.025,
            "central_soft_crop_98": 0.01,
            "central_soft_crop_99": 0.005,
            "central_soft_crop_90": 0.05,
            "central_soft_crop_85": 0.075,
            "central_soft_crop_80": 0.10,
            "central_soft_crop_75": 0.125,
        }
        if mode not in {*crop_modes, "dual_view"}:
            raise ValueError(f"Unsupported enhanced encoder mode: {mode}")
        self.crop_margin = crop_modes.get(mode, 0.125)
        self.base_encoder = base_encoder
        self.mode = mode
        self.adapter = DualViewAdapter()
        self.fusion_scale = nn.Parameter(torch.zeros(()))

    def _central(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        y0, x0 = int(height * self.crop_margin), int(width * self.crop_margin)
        cropped = images[..., y0 : height - y0, x0 : width - x0]
        return torch.nn.functional.interpolate(
            cropped, size=(height, width), mode="bilinear", align_corners=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must be a four-dimensional tensor")
        full = self.base_encoder(images)
        center = self.base_encoder(self._central(images))
        residual = center - full.detach()
        crop_delta = self.adapter(residual) - residual
        if self.mode in {"central_soft_crop", "central_soft_crop_95"}:
            return full + crop_delta
        if self.mode in {"central_soft_crop_99", "dual_view"}:
            return full + torch.sigmoid(self.fusion_scale) * crop_delta
        full_delta = self.adapter.full_adapter(full) - full
        return full + crop_delta + full_delta

class AttendedDualViewEncoder(nn.Module):
    """Compose neutral residual CBAM with the dual-view residual adapter."""

    output_dimension = 512

    def __init__(self, attention_encoder: AttentionResNet18Encoder, mode: str = "central_soft_crop") -> None:
        super().__init__()
        self.attention_encoder = attention_encoder
        self.dual_encoder = DualViewEncoder(attention_encoder, mode=mode)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.dual_encoder(images)

class EnhancedProtoNet(nn.Module):
    """Composition wrapper preserving the standard ProtoNet episode interface."""

    def __init__(self, encoder: nn.Module, *, way: int = 3) -> None:
        super().__init__()
        from .protonet import ProtoNet
        self.encoder = encoder
        self.way = int(way)
        self.head = ProtoNet(encoder, way=way)

    @property
    def encoder_module(self) -> nn.Module:
        return self.encoder

    def forward(self, support_images, support_labels, query_images, query_labels=None, episode_class_ids=None):
        return self.head(support_images, support_labels, query_images, query_labels, episode_class_ids=episode_class_ids)

def _freeze_backbone_but_enable_modules(encoder: nn.Module, backbone: nn.Module) -> None:
    """Keep an M2 backbone frozen while allowing neutral enhancement modules."""
    encoder.requires_grad_(False)
    for module in encoder.modules():
        if isinstance(module, (AttentionAdapter, type(backbone.attention))):
            for parameter in module.parameters(recurse=True):
                parameter.requires_grad_(True)
    original_train = nn.Module.train

    def frozen_train(mode: bool = True):
        original_train(encoder, mode)
        original_train(backbone.backbone, False)
        if hasattr(encoder, "dual_encoder") and hasattr(encoder.dual_encoder, "adapter"):
            original_train(encoder.dual_encoder.adapter, mode)
        return encoder

    encoder.train = frozen_train

def _frozen_reference_train(encoder: nn.Module):
    original_train = nn.Module.train
    def frozen_reference_train(mode: bool = True):
        original_train(encoder, False)
        return encoder
    return frozen_reference_train

def build_enhanced_encoder(
    *,
    roi_mode: str,
    pretrain_mode: str,
    attention_mode: str,
    m2_checkpoint: object | None = None,
    attention_type: str = "residual_cbam",
    reduction_ratio: int = 16,
    spatial_kernel_size: int = 7,
) -> nn.Module:
    """Construct the encoder with the requested pretraining and ablation settings."""
    if roi_mode not in {
        "none", "central_soft_crop", "central_soft_crop_95",
        "central_soft_crop_98", "central_soft_crop_99",
        "central_soft_crop_90", "central_soft_crop_85",
        "central_soft_crop_80", "central_soft_crop_75", "dual_view",
    }:
        raise ValueError(f"Unsupported ROI mode: {roi_mode}")
    if pretrain_mode not in {"imagenet", "m2_frozen", "m2_finetune"}:
        raise ValueError(f"Unsupported pretraining mode: {pretrain_mode}")

    # Handle M2 finetune: load M2 weights but keep backbone trainable
    m2_finetune = pretrain_mode == "m2_finetune"
    m2_frozen = pretrain_mode == "m2_frozen"

    if m2_frozen or m2_finetune:
        if m2_checkpoint is None:
            raise ValueError("m2 mode requires an M2 checkpoint")
        base = ResNet18SpatialEncoder(pretrained=False, frozen=m2_frozen)
        m2_checkpoint(base)
    else:
        base = ResNet18SpatialEncoder(pretrained=True, frozen=False)

    if attention_mode == "residual_cbam":
        spatial = AttentionResNet18Encoder(
            pretrained=False, frozen=False,
            attention_type=attention_type,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
        )
        spatial.backbone.fc = nn.Identity()
        spatial.backbone.load_state_dict(base.backbone.state_dict(), strict=False)
        if roi_mode == "none":
            encoder = spatial
        else:
            encoder = AttendedDualViewEncoder(spatial, mode=roi_mode)
        if m2_frozen:
            _freeze_backbone_but_enable_modules(encoder, spatial)
        return encoder

    if roi_mode != "none":
        flat_encoder = ResNet18Encoder(pretrained=False, frozen=m2_frozen)
        flat_encoder.backbone.load_state_dict(base.backbone.state_dict(), strict=False)
        if m2_frozen:
            flat_encoder.requires_grad_(False)
            for module in flat_encoder.modules():
                if isinstance(module, AttentionAdapter):
                    for parameter in module.parameters(recurse=False):
                        parameter.requires_grad_(True)
            original_train = nn.Module.train
            def dual_frozen_train(mode: bool = True):
                original_train(flat_encoder, mode)
                original_train(flat_encoder.backbone, False)
                return flat_encoder
            flat_encoder.train = dual_frozen_train
        return DualViewEncoder(flat_encoder, mode=roi_mode)

    flat_encoder = ResNet18Encoder(pretrained=False, frozen=m2_frozen)
    flat_encoder.backbone.load_state_dict(base.backbone.state_dict(), strict=False)
    if m2_frozen:
        flat_encoder.requires_grad_(False)
        flat_encoder.train = _frozen_reference_train(flat_encoder)
    return flat_encoder
