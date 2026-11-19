"""M2 supervised-transfer protocol and episodic validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional

from .models.logistic import fit_multinomial_l2_logistic_regression
from .training import episode_macro_f1


@dataclass(frozen=True)
class SupervisedTransferProtocol:
    learning_rate_candidates: tuple[float, ...]
    adam_betas: tuple[float, float]
    adam_epsilon: float
    weight_decay: float
    batch_size: int
    maximum_epochs: int
    validation_interval_epochs: int
    minimum_epochs: int
    early_stopping_patience_checks: int
    regularization_c_candidates: tuple[float, ...]
    logistic_maximum_iterations: int
    logistic_tolerance: float
    selection_split_id: str
    mixed_precision_on_cuda: bool
    resource_monitor_interval_epochs: int
    training_seed: int
    supplementary_seed_split: str
    supplementary_training_seeds: tuple[int, ...]


@dataclass(frozen=True)
class M2ValidationResult:
    regularization_c: float
    mean_accuracy: float
    mean_macro_f1: float
    mean_loss: float
    episode_count: int
    query_count: int
    shot_metrics: dict[int, dict[str, float | int]]
    prediction_rows: tuple[dict[str, object], ...]

    @property
    def selection_key(self) -> tuple[float, float, float]:
        return self.mean_accuracy, self.mean_macro_f1, -self.mean_loss


class SupervisedTransferModel(nn.Module):
    """A trainable ResNet18 encoder and temporary base-disease classifier."""

    def __init__(self, encoder: nn.Module, number_of_classes: int) -> None:
        super().__init__()
        if number_of_classes < 2:
            raise ValueError("number_of_classes must be at least two")
        output_dimension = getattr(encoder, "output_dimension", None)
        if output_dimension is None:
            raise ValueError("encoder must declare output_dimension")
        self.encoder = encoder
        self.classifier = nn.Linear(int(output_dimension), number_of_classes)
        self.number_of_classes = int(number_of_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))


def load_supervised_transfer_protocol(path: Path) -> SupervisedTransferProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["model_id"] != "M2":
        raise ValueError("Configuration is not for M2")
    if payload["model_name"] != "SupervisedTransferLogistic":
        raise ValueError("Unexpected M2 model name")
    if payload["backbone"] != "resnet18_imagenet1k_v1":
        raise ValueError("M2 must start from ImageNet-1K V1 ResNet18")
    if not payload["fine_tune_backbone"]:
        raise ValueError("M2 must fine-tune the complete encoder")
    sampling = payload["supervised_sampling"]
    if sampling != {
        "class_sampling": "uniform_balanced_cycles",
        "patient_sampling": "uniform_without_replacement_cycles",
        "image_sampling": "one_uniform_image_per_selected_patient",
        "samples_per_epoch": "total_unique_training_patients",
    }:
        raise ValueError("Unsupported M2 supervised sampling protocol")
    optimizer = payload["optimizer"]
    if optimizer["name"] != "Adam":
        raise ValueError("M2 must use Adam")
    if payload["learning_rate_schedule"] != "constant":
        raise ValueError("Only the approved constant learning rate is supported")
    if payload["gradient_clipping"] is not None:
        raise ValueError("Gradient clipping is not part of the approved M2 protocol")
    head = payload["episode_classifier"]
    if head["name"] != "multinomial_l2_logistic_regression":
        raise ValueError("M2 must use multinomial L2 logistic regression")
    if not head["fit_intercept"] or head["feature_l2_normalization"]:
        raise ValueError("M2 requires an intercept and unnormalised features")
    if head["class_weights"] is not None:
        raise ValueError("M2 does not use class weights")
    selection = payload["hyperparameter_selection"]
    if not selection["common_across_shots"]:
        raise ValueError("M2 hyperparameters must be common to 3-shot and 5-shot")
    stability = payload["seed_stability"]
    if stability["analysis_role"] != "supplementary":
        raise ValueError("Seed stability must remain supplementary")

    rates = tuple(float(value) for value in optimizer["learning_rate_candidates"])
    if rates != (0.0001, 0.0003):
        raise ValueError("Approved M2 learning rates are 1e-4 and 3e-4")
    c_values = tuple(float(value) for value in head["regularization_c_candidates"])
    if c_values != (0.01, 0.1, 1.0, 10.0):
        raise ValueError("Approved M2 C candidates are 0.01, 0.1, 1, and 10")
    betas = tuple(float(value) for value in optimizer["betas"])
    seeds = tuple(int(value) for value in stability["training_seeds"])
    primary_seed = int(payload["training_seed"])
    if len(betas) != 2:
        raise ValueError("Adam requires exactly two beta values")
    if len(seeds) != 3 or len(set(seeds)) != 3 or primary_seed not in seeds:
        raise ValueError("Seed analysis requires three unique seeds including the main seed")

    protocol = SupervisedTransferProtocol(
        learning_rate_candidates=rates,
        adam_betas=(betas[0], betas[1]),
        adam_epsilon=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        batch_size=int(payload["batch_size"]),
        maximum_epochs=int(payload["maximum_epochs"]),
        validation_interval_epochs=int(payload["validation_interval_epochs"]),
        minimum_epochs=int(payload["minimum_epochs"]),
        early_stopping_patience_checks=int(
            payload["early_stopping_patience_checks"]
        ),
        regularization_c_candidates=c_values,
        logistic_maximum_iterations=int(head["maximum_iterations"]),
        logistic_tolerance=float(head["tolerance"]),
        selection_split_id=str(selection["split_id"]),
        mixed_precision_on_cuda=bool(payload["mixed_precision_on_cuda"]),
        resource_monitor_interval_epochs=int(
            payload["resource_monitor_interval_epochs"]
        ),
        training_seed=primary_seed,
        supplementary_seed_split=str(stability["split_id"]),
        supplementary_training_seeds=seeds,
    )
    if protocol.batch_size != 32:
        raise ValueError("Approved M2 batch size is 32")
    if protocol.resource_monitor_interval_epochs < 1:
        raise ValueError("M2 resource monitor interval must be positive")
    if (
        protocol.maximum_epochs != 50
        or protocol.validation_interval_epochs != 5
        or protocol.minimum_epochs != 10
        or protocol.early_stopping_patience_checks != 4
    ):
        raise ValueError("M2 epoch and early-stopping controls differ from approval")
    return protocol


def resolve_m2_training_seed(
    protocol: SupervisedTransferProtocol,
    split_id: str,
    requested_seed: int | None,
) -> int:
    seed = protocol.training_seed if requested_seed is None else int(requested_seed)
    if seed == protocol.training_seed:
        return seed
    if (
        split_id == protocol.supplementary_seed_split
        and seed in protocol.supplementary_training_seeds
    ):
        return seed
    raise ValueError("Non-primary seeds are restricted to split_01 supplementary runs")


def _move_episode_features(
    encoder: nn.Module,
    episode: dict[str, object],
    device: torch.device,
    mixed_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support_images = episode["support_images"].to(
        device, non_blocking=device.type == "cuda"
    )
    query_images = episode["query_images"].to(
        device, non_blocking=device.type == "cuda"
    )
    support_labels = episode["support_labels"].to(
        device, non_blocking=device.type == "cuda"
    )
    query_labels = episode["query_labels"].to(
        device, non_blocking=device.type == "cuda"
    )
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=mixed_precision
    ):
        support_features = encoder(support_images).float()
        query_features = encoder(query_images).float()
    return support_features, support_labels, query_features, query_labels


def evaluate_m2_validation(
    encoder: nn.Module,
    episodes_by_shot: Mapping[int, Iterable[dict[str, object]]],
    regularization_values: Sequence[float],
    *,
    device: torch.device,
    mixed_precision: bool,
    maximum_iterations: int = 100,
    tolerance: float = 1e-7,
) -> tuple[M2ValidationResult, ...]:
    """Evaluate all C values while sharing each ResNet18 feature extraction."""

    shots = tuple(sorted(int(shot) for shot in episodes_by_shot))
    if shots != (3, 5):
        raise ValueError("M2 validation requires both 3-shot and 5-shot episodes")
    c_values = tuple(float(value) for value in regularization_values)
    if not c_values or len(set(c_values)) != len(c_values):
        raise ValueError("regularization_values must be non-empty and unique")
    encoder.to(device)
    encoder.eval()
    accumulator: dict[float, dict[int, dict[str, object]]] = {
        value: {
            shot: {"accuracies": [], "f1": [], "losses": [], "outputs": []}
            for shot in shots
        }
        for value in c_values
    }
    episode_metadata: dict[int, list[dict[str, object]]] = {
        shot: [] for shot in shots
    }

    for shot in shots:
        episode_count = 0
        for episode in episodes_by_shot[shot]:
            episode_count += 1
            support_features, support_labels, query_features, query_labels = (
                _move_episode_features(
                    encoder, episode, device, mixed_precision
                )
            )
            way = len(tuple(episode["class_ids"]))
            episode_metadata[shot].append(
                {
                    "episode_id": episode["episode_id"],
                    "split_id": episode["split_id"],
                    "query_metadata": tuple(episode["query_metadata"]),
                }
            )
            for c_value in c_values:
                fit = fit_multinomial_l2_logistic_regression(
                    support_features,
                    support_labels,
                    way=way,
                    regularization_c=c_value,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
                logits = functional.linear(query_features, fit.weights, fit.bias)
                probabilities = functional.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)
                losses = functional.cross_entropy(
                    logits, query_labels, reduction="none"
                )
                target = accumulator[c_value][shot]
                target["accuracies"].append(
                    float((predictions == query_labels).float().mean().cpu())
                )
                target["f1"].append(
                    episode_macro_f1(
                        predictions.detach().cpu(), query_labels.detach().cpu(), way
                    )
                )
                target["losses"].append(float(losses.mean().cpu()))
                cpu_predictions = predictions.detach().cpu()
                cpu_labels = query_labels.detach().cpu()
                cpu_probabilities = probabilities.detach().cpu()
                cpu_logits = logits.detach().cpu()
                cpu_losses = losses.detach().cpu()
                target["outputs"].append(
                    (
                        cpu_predictions,
                        cpu_labels,
                        cpu_probabilities,
                        cpu_logits,
                        cpu_losses,
                    )
                )
        if episode_count == 0:
            raise ValueError(f"Validation loader for {shot}-shot is empty")

    results: list[M2ValidationResult] = []
    for c_value in c_values:
        shot_metrics: dict[int, dict[str, float | int]] = {}
        query_count = 0
        for shot in shots:
            item = accumulator[c_value][shot]
            accuracies = item["accuracies"]
            f1_scores = item["f1"]
            losses = item["losses"]
            shot_query_count = sum(
                int(output[1].numel()) for output in item["outputs"]
            )
            query_count += shot_query_count
            shot_metrics[shot] = {
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "mean_macro_f1": sum(f1_scores) / len(f1_scores),
                "mean_loss": sum(losses) / len(losses),
                "episode_count": len(accuracies),
                "query_count": shot_query_count,
            }
        results.append(
            M2ValidationResult(
                regularization_c=c_value,
                mean_accuracy=sum(
                    float(shot_metrics[shot]["mean_accuracy"]) for shot in shots
                )
                / len(shots),
                mean_macro_f1=sum(
                    float(shot_metrics[shot]["mean_macro_f1"]) for shot in shots
                )
                / len(shots),
                mean_loss=sum(
                    float(shot_metrics[shot]["mean_loss"]) for shot in shots
                )
                / len(shots),
                episode_count=sum(
                    int(shot_metrics[shot]["episode_count"]) for shot in shots
                ),
                query_count=query_count,
                shot_metrics=shot_metrics,
                prediction_rows=(),
            )
        )
    best = select_best_m2_validation(results)
    best_rows: list[dict[str, object]] = []
    for shot in shots:
        compact_outputs = accumulator[best.regularization_c][shot]["outputs"]
        for metadata_record, output in zip(
            episode_metadata[shot], compact_outputs, strict=True
        ):
            (
                cpu_predictions,
                cpu_labels,
                cpu_probabilities,
                cpu_logits,
                cpu_losses,
            ) = output
            for query_index, metadata in enumerate(
                metadata_record["query_metadata"]
            ):
                best_rows.append(
                    {
                        "episode_id": metadata_record["episode_id"],
                        "split_id": metadata_record["split_id"],
                        "shot": shot,
                        "regularization_c": best.regularization_c,
                        "query_index": query_index,
                        "class_id": metadata["class_id"],
                        "patient_id": metadata["patient_id"],
                        "image_id": metadata["image_id"],
                        "source_relpath": metadata["source_relpath"],
                        "target_label": int(cpu_labels[query_index]),
                        "predicted_label": int(cpu_predictions[query_index]),
                        "correct": bool(
                            cpu_predictions[query_index]
                            == cpu_labels[query_index]
                        ),
                        "confidence": float(
                            cpu_probabilities[query_index].max()
                        ),
                        "query_loss": float(cpu_losses[query_index]),
                        "logits": cpu_logits[query_index].tolist(),
                        "probabilities": cpu_probabilities[query_index].tolist(),
                    }
                )
    return tuple(
        replace(result, prediction_rows=tuple(best_rows))
        if result is best
        else result
        for result in results
    )


def select_best_m2_validation(
    results: Sequence[M2ValidationResult],
) -> M2ValidationResult:
    if not results:
        raise ValueError("No M2 validation candidates were supplied")
    return max(results, key=lambda result: result.selection_key)
