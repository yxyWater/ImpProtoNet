"""Approved M3 MatchingNet protocol, selection, and checkpoint loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .data.episodes import load_episode_protocol
from .models import MatchingNet, build_resnet18_encoder
from .training import ProtoNetTrainingProtocol, select_common_learning_rate


@dataclass(frozen=True, kw_only=True)
class MatchingNetTrainingProtocol(ProtoNetTrainingProtocol):
    processing_steps: int
    encoders_shared: bool
    support_order_seed: int
    cosine_temperature: float
    probability_epsilon: float


def load_matchingnet_training_protocol(
    training_config_path: Path,
    episode_protocol_path: Path,
) -> MatchingNetTrainingProtocol:
    payload = json.loads(training_config_path.read_text(encoding="utf-8"))
    episodes = load_episode_protocol(episode_protocol_path)
    if payload["model_id"] != "M3" or payload["model_name"] != "MatchingNetFCE":
        raise ValueError("Configuration is not for M3 MatchingNetFCE")
    if payload["backbone"] != "resnet18_imagenet1k_v1":
        raise ValueError("M3 must use ImageNet-1K V1 ResNet18")
    if not payload["fine_tune_backbone"]:
        raise ValueError("M3 must fine-tune the complete backbone")
    if not payload["support_query_encoder_shared"]:
        raise ValueError("M3 requires the confirmed shared support/query encoder")
    context = payload["full_context_embedding"]
    expected_context = {
        "enabled": True,
        "support_embedding": "bidirectional_lstm_residual_sum",
        "query_embedding": "augmented_hidden_attention_lstm_residual",
        "query_processing_steps": 5,
    }
    if context != expected_context:
        raise ValueError("M3 full-context embedding differs from the confirmed design")
    matching = payload["matching"]
    if matching != {
        "similarity": "cosine",
        "temperature": 1.0,
        "learnable_temperature": False,
        "class_probability": "sum_support_attention_by_one_hot_label",
        "probability_epsilon": 1e-12,
    }:
        raise ValueError("M3 matching rule differs from the original-paper design")
    support_order = payload["support_order"]
    if support_order != {
        "method": "deterministic_sha256_pseudorandom",
        "seed": 20261004,
        "identity_fields": ["episode_id", "patient_id", "image_id"],
        "preserve_nested_relative_order": True,
    }:
        raise ValueError("M3 support ordering differs from the confirmed design")
    optimizer = payload["optimizer"]
    if optimizer["name"] != "Adam":
        raise ValueError("M3 must use Adam")
    candidates = tuple(float(value) for value in optimizer["learning_rate_candidates"])
    if candidates != (0.0001, 0.0003):
        raise ValueError("Approved M3 learning rates are 1e-4 and 3e-4")
    betas = tuple(float(value) for value in optimizer["betas"])
    if len(betas) != 2:
        raise ValueError("Adam requires exactly two beta values")
    selection = payload["learning_rate_selection"]
    if not selection["common_across_shots"]:
        raise ValueError("One M3 learning rate must be shared across shots")
    if payload["learning_rate_schedule"] != "constant":
        raise ValueError("Only the approved constant learning rate is supported")
    if payload["gradient_clipping"] is not None:
        raise ValueError("Gradient clipping is not part of the approved M3 protocol")
    stability = payload["seed_stability"]
    seeds = tuple(int(value) for value in stability["training_seeds"])
    primary_seed = int(payload["training_seed"])
    if (
        stability["analysis_role"] != "supplementary"
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or primary_seed not in seeds
    ):
        raise ValueError("M3 seed stability configuration is invalid")
    protocol = MatchingNetTrainingProtocol(
        learning_rate_candidates=candidates,
        adam_betas=(betas[0], betas[1]),
        adam_epsilon=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        learning_rate_selection_split=str(selection["split_id"]),
        mixed_precision_on_cuda=bool(payload["mixed_precision_on_cuda"]),
        resource_monitor_interval_episodes=int(
            payload["resource_monitor_interval_episodes"]
        ),
        training_seed=primary_seed,
        supplementary_seed_split=str(stability["split_id"]),
        supplementary_training_seeds=seeds,
        validation_interval_episodes=episodes.validation_interval_episodes,
        minimum_train_episodes=episodes.minimum_train_episodes,
        early_stopping_patience_checks=episodes.early_stopping_patience_checks,
        maximum_train_episodes=episodes.maximum_train_episodes,
        processing_steps=int(context["query_processing_steps"]),
        encoders_shared=bool(payload["support_query_encoder_shared"]),
        support_order_seed=int(support_order["seed"]),
        cosine_temperature=float(matching["temperature"]),
        probability_epsilon=float(matching["probability_epsilon"]),
    )
    if protocol.resource_monitor_interval_episodes < 1:
        raise ValueError("M3 resource monitor interval must be positive")
    return protocol


def build_matchingnet_model(
    protocol: MatchingNetTrainingProtocol,
    *,
    pretrained: bool,
) -> MatchingNet:
    if not protocol.encoders_shared:
        raise ValueError("This experiment requires one shared ResNet18 encoder")
    return MatchingNet(
        build_resnet18_encoder(pretrained=pretrained),
        way=3,
        processing_steps=protocol.processing_steps,
        probability_epsilon=protocol.probability_epsilon,
    )


def select_matchingnet_learning_rate(
    summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    if any(summary.get("model_id") != "M3" for summary in summaries):
        raise ValueError("MatchingNet selection only accepts M3 summaries")
    result = select_common_learning_rate(summaries)
    return {"model_id": "M3", "model_name": "MatchingNetFCE", **result}


def load_best_matchingnet_checkpoint(
    model: MatchingNet,
    checkpoint_path: Path,
    *,
    expected_split_id: str,
    expected_shot: int,
    expected_training_seed: int,
    expected_learning_rate: float | None = None,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_id") != "M3" or payload.get("model_name") != "MatchingNetFCE":
        raise ValueError("Checkpoint is not an M3 MatchingNetFCE checkpoint")
    if payload.get("checkpoint_purpose") != "inference_best":
        raise ValueError("Final M3 evaluation requires the validation-best checkpoint")
    if payload.get("split_id") != expected_split_id:
        raise ValueError("M3 checkpoint belongs to a different split")
    if int(payload.get("shot", -1)) != expected_shot:
        raise ValueError("M3 checkpoint belongs to a different shot")
    if int(payload.get("training_seed", -1)) != expected_training_seed:
        raise ValueError("M3 checkpoint uses a different training seed")
    if expected_learning_rate is not None and not math.isclose(
        float(payload["learning_rate"]), expected_learning_rate
    ):
        raise ValueError("M3 checkpoint uses a different learning rate")
    if payload.get("best_validation") is None:
        raise ValueError("M3 checkpoint has no validation-selected state")
    model.load_state_dict(payload["model_state"])
    return payload
