"""Approved M5 spatial RelationNet protocol and checkpoint handling."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .data.episodes import load_episode_protocol
from .models import RelationNet, build_resnet18_spatial_encoder
from .training import ProtoNetTrainingProtocol, select_common_learning_rate


@dataclass(frozen=True, kw_only=True)
class RelationNetTrainingProtocol(ProtoNetTrainingProtocol):
    first_hidden_channels: int
    second_hidden_channels: int
    fully_connected_hidden: int
    probability_epsilon: float
    support_aggregation: str


def load_relationnet_training_protocol(
    training_config_path: Path,
    episode_protocol_path: Path,
) -> RelationNetTrainingProtocol:
    payload = json.loads(training_config_path.read_text(encoding="utf-8"))
    episodes = load_episode_protocol(episode_protocol_path)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported M5 configuration schema")
    if payload.get("model_id") != "M5" or payload.get("model_name") != "RelationNetSpatial":
        raise ValueError("Configuration is not for M5 RelationNetSpatial")
    if payload.get("backbone") != "resnet18_imagenet1k_v1_spatial_layer4":
        raise ValueError("M5 must use the approved spatial ImageNet ResNet18")
    if not payload.get("fine_tune_backbone"):
        raise ValueError("M5 must fine-tune the complete backbone")
    if payload.get("spatial_feature_shape_at_224") != [512, 7, 7]:
        raise ValueError("M5 spatial feature declaration must be 512x7x7")
    if payload.get("support_aggregation") != "elementwise_sum_by_class":
        raise ValueError("M5 must use original-paper class-wise feature-map sums")
    relation = payload["relation_module"]
    if relation != {
        "pair_combination": "channel_concatenation",
        "first_hidden_channels": 256,
        "second_hidden_channels": 64,
        "fully_connected_hidden": 8,
        "relation_score_activation": "sigmoid",
    }:
        raise ValueError("M5 relation module differs from the confirmed design")
    if payload["objective"] != {
        "name": "mean_squared_error",
        "targets": "one_hot_class_match",
        "reduction": "mean_over_classes_then_queries",
    }:
        raise ValueError("M5 must use the confirmed original-paper MSE objective")
    prediction = payload["prediction"]
    if prediction != {
        "class_rule": "maximum_relation_score",
        "probability_rule": "normalize_sigmoid_relation_scores_by_sum",
        "probability_epsilon": 1e-12,
        "retain_raw_relation_scores": True,
        "retain_pre_sigmoid_logits": True,
    }:
        raise ValueError("M5 probability and raw-output rules differ from the confirmed design")
    optimizer = payload["optimizer"]
    if optimizer.get("name") != "Adam":
        raise ValueError("M5 must use Adam")
    candidates = tuple(float(value) for value in optimizer["learning_rate_candidates"])
    if candidates != (0.0001, 0.0003):
        raise ValueError("Approved M5 learning rates are 1e-4 and 3e-4")
    betas = tuple(float(value) for value in optimizer["betas"])
    if len(betas) != 2:
        raise ValueError("Adam requires exactly two beta values")
    selection = payload["learning_rate_selection"]
    if not selection.get("common_across_shots"):
        raise ValueError("One M5 learning rate must be shared across shots")
    if payload.get("learning_rate_schedule") != "constant":
        raise ValueError("Only the approved constant learning rate is supported")
    if payload.get("gradient_clipping") is not None:
        raise ValueError("Gradient clipping is not part of the approved M5 protocol")
    stability = payload["seed_stability"]
    seeds = tuple(int(value) for value in stability["training_seeds"])
    primary_seed = int(payload["training_seed"])
    if (
        stability.get("analysis_role") != "supplementary"
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or primary_seed not in seeds
    ):
        raise ValueError("M5 seed stability configuration is invalid")
    protocol = RelationNetTrainingProtocol(
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
        first_hidden_channels=int(relation["first_hidden_channels"]),
        second_hidden_channels=int(relation["second_hidden_channels"]),
        fully_connected_hidden=int(relation["fully_connected_hidden"]),
        probability_epsilon=float(prediction["probability_epsilon"]),
        support_aggregation=str(payload["support_aggregation"]),
    )
    if protocol.resource_monitor_interval_episodes < 1:
        raise ValueError("M5 resource monitor interval must be positive")
    return protocol


def build_relationnet_model(
    protocol: RelationNetTrainingProtocol,
    *,
    pretrained: bool,
) -> RelationNet:
    return RelationNet(
        build_resnet18_spatial_encoder(pretrained=pretrained),
        way=3,
        first_hidden_channels=protocol.first_hidden_channels,
        second_hidden_channels=protocol.second_hidden_channels,
        fully_connected_hidden=protocol.fully_connected_hidden,
        probability_epsilon=protocol.probability_epsilon,
    )


def select_relationnet_learning_rate(
    summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    if any(summary.get("model_id") != "M5" for summary in summaries):
        raise ValueError("RelationNet selection only accepts M5 summaries")
    result = select_common_learning_rate(summaries)
    return {"model_id": "M5", "model_name": "RelationNetSpatial", **result}


def load_best_relationnet_checkpoint(
    model: RelationNet,
    checkpoint_path: Path,
    *,
    expected_split_id: str,
    expected_shot: int,
    expected_training_seed: int,
    expected_learning_rate: float | None = None,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_id") != "M5" or payload.get("model_name") != "RelationNetSpatial":
        raise ValueError("Checkpoint is not an M5 RelationNetSpatial checkpoint")
    if payload.get("checkpoint_purpose") != "inference_best":
        raise ValueError("Final M5 evaluation requires the validation-best checkpoint")
    if payload.get("split_id") != expected_split_id:
        raise ValueError("M5 checkpoint belongs to a different split")
    if int(payload.get("shot", -1)) != expected_shot:
        raise ValueError("M5 checkpoint belongs to a different shot")
    if int(payload.get("training_seed", -1)) != expected_training_seed:
        raise ValueError("M5 checkpoint uses a different training seed")
    if expected_learning_rate is not None and not math.isclose(
        float(payload["learning_rate"]), expected_learning_rate
    ):
        raise ValueError("M5 checkpoint uses a different learning rate")
    if payload.get("best_validation") is None:
        raise ValueError("M5 checkpoint has no validation-selected state")
    model.load_state_dict(payload["model_state"])
    return payload

