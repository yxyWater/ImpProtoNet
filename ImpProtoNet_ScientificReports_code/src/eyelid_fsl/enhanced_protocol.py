"""Configuration and model construction for the enhanced ProtoNet study.

All variants remain leakage-free: every trainable module sees only training
images during training, and model/layer selection uses validation episodes only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .data.episodes import load_episode_protocol
from .training import ProtoNetTrainingProtocol


@dataclass(frozen=True)
class EnhancedSpec:
    model_id: str
    model_name: str
    roi_mode: str
    pretrain_mode: str
    attention_mode: str
    sampling_mode: str
    sampling_variant: str
    training: ProtoNetTrainingProtocol

    def layer_id(self) -> str:
        roi_ids = {
            "none": "R0",
            "central_soft_crop": "R1",
            "central_soft_crop_95": "R95",
            "central_soft_crop_98": "R98",
            "central_soft_crop_99": "R99",
            "central_soft_crop_90": "R90",
            "central_soft_crop_85": "R85",
            "central_soft_crop_80": "R80",
            "central_soft_crop_75": "R75",
            "dual_view": "RD",
        }
        if self.roi_mode not in roi_ids:
            raise ValueError(f"Unsupported ROI mode for layer id: {self.roi_mode}")
        parts = [
            "E0",
            roi_ids[self.roi_mode],
            "P1" if self.pretrain_mode != "imagenet" else "P0",
            "A1" if self.attention_mode != "none" else "A0",
            "S0" if self.sampling_mode == "standard" else (self.sampling_variant or "S1"),
        ]
        return "".join(parts)

    def experiment_metadata(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id(),
            "roi_mode": self.roi_mode,
            "pretrain_mode": self.pretrain_mode,
            "attention_mode": self.attention_mode,
            "sampling_mode": self.sampling_mode,
            "sampling_variant": self.sampling_variant,
            "selection_partition": "validation",
            "test_opened": False,
            "analysis_role": "enhanced_protonet_screening",
        }


def load_enhanced_spec(
    training_config_path: Path,
    episode_protocol_path: Path,
    *,
    roi_mode: str,
    pretrain_mode: str,
    attention_mode: str,
    sampling_mode: str,
    sampling_variant: str,
) -> EnhancedSpec:
    payload = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    episode_protocol = load_episode_protocol(episode_protocol_path)
    optimizer = payload["optimizer"]
    seed_stability = payload["seed_stability"]
    candidates = tuple(float(value) for value in optimizer["learning_rate_candidates"])
    betas = tuple(float(value) for value in optimizer["betas"])
    training = ProtoNetTrainingProtocol(
        learning_rate_candidates=candidates,
        adam_betas=(betas[0], betas[1]),
        adam_epsilon=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        learning_rate_selection_split=str(payload["learning_rate_selection"]["split_id"]),
        mixed_precision_on_cuda=bool(payload["mixed_precision_on_cuda"]),
        resource_monitor_interval_episodes=int(payload["resource_monitor_interval_episodes"]),
        training_seed=int(payload["training_seed"]),
        supplementary_seed_split=str(seed_stability["split_id"]),
        supplementary_training_seeds=tuple(int(v) for v in seed_stability["training_seeds"]),
        validation_interval_episodes=episode_protocol.validation_interval_episodes,
        minimum_train_episodes=episode_protocol.minimum_train_episodes,
        early_stopping_patience_checks=episode_protocol.early_stopping_patience_checks,
        maximum_train_episodes=episode_protocol.maximum_train_episodes,
    )
    for key, choices in {
        "roi_mode": (
            roi_mode,
            (
                "none",
                "central_soft_crop",
                "central_soft_crop_95",
                "central_soft_crop_98",
                "central_soft_crop_99",
                "central_soft_crop_90",
                "central_soft_crop_85",
                "central_soft_crop_80", "central_soft_crop_75",
                "dual_view",
            ),
        ),
        "pretrain_mode": (pretrain_mode, ("imagenet", "m2_frozen", "m2_finetune")),
        "attention_mode": (attention_mode, ("none", "residual_cbam")),
        "sampling_mode": (sampling_mode, ("standard", "patient_exposure_balanced")),
    }.items():
        if choices[0] not in choices[1]:
            raise ValueError(f"Unsupported {key}: {choices[0]}")
    return EnhancedSpec(
        model_id="ENH",
        model_name="EnhancedProtoNet",
        roi_mode=roi_mode,
        pretrain_mode=pretrain_mode,
        attention_mode=attention_mode,
        sampling_mode=sampling_mode,
        sampling_variant=sampling_variant,
        training=training,
    )
