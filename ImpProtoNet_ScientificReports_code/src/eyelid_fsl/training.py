"""Training, validation, logging, and checkpointing for M0 ProtoNet."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torchvision
from .data.episodes import load_episode_protocol


@dataclass(frozen=True)
class ProtoNetTrainingProtocol:
    learning_rate_candidates: tuple[float, ...]
    adam_betas: tuple[float, float]
    adam_epsilon: float
    weight_decay: float
    learning_rate_selection_split: str
    mixed_precision_on_cuda: bool
    resource_monitor_interval_episodes: int
    training_seed: int
    supplementary_seed_split: str
    supplementary_training_seeds: tuple[int, ...]
    validation_interval_episodes: int
    minimum_train_episodes: int
    early_stopping_patience_checks: int
    maximum_train_episodes: int
    lr_schedule: str = "constant"


@dataclass(frozen=True)
class ValidationResult:
    mean_accuracy: float
    mean_macro_f1: float
    mean_loss: float
    episode_count: int
    query_count: int
    prediction_rows: tuple[dict[str, object], ...]


def load_protonet_training_protocol(
    training_config_path: Path,
    episode_protocol_path: Path,
) -> ProtoNetTrainingProtocol:
    payload = json.loads(training_config_path.read_text(encoding="utf-8"))
    episode_protocol = load_episode_protocol(episode_protocol_path)
    optimizer = payload["optimizer"]
    selection = payload["learning_rate_selection"]
    seed_stability = payload["seed_stability"]
    if payload["model_id"] != "M0" or payload["model_name"] != "ProtoNet":
        raise ValueError("Training configuration is not for M0 ProtoNet")
    if payload["backbone"] != "resnet18_imagenet1k_v1":
        raise ValueError("M0 must use the approved ImageNet ResNet18 backbone")
    if not payload["fine_tune_backbone"]:
        raise ValueError("M0 must fine-tune the complete backbone")
    if optimizer["name"] != "Adam":
        raise ValueError("M0 must use the approved Adam optimizer")
    if payload["learning_rate_schedule"] not in ("constant", "cosine"):
        raise ValueError("Only the approved constant learning rate is supported")
    if payload["gradient_clipping"] is not None:
        raise ValueError("Gradient clipping is not part of the approved M0 protocol")
    if not selection["common_across_shots"]:
        raise ValueError("One learning rate must be shared by 3-shot and 5-shot")
    if seed_stability["analysis_role"] != "supplementary":
        raise ValueError("Training-seed stability must remain a supplementary analysis")

    candidates = tuple(float(value) for value in optimizer["learning_rate_candidates"])
    if not set(candidates).issubset({0.00005, 0.0001, 0.0003, 0.0005, 0.001}):
        raise ValueError(
            "Approved learning-rate candidates are 0.00005, 0.0001, 0.0003, 0.0005, and 0.001"
        )
    betas = tuple(float(value) for value in optimizer["betas"])
    if len(betas) != 2:
        raise ValueError("Adam requires two beta values")
    training_seed = int(payload["training_seed"])
    supplementary_training_seeds = tuple(
        int(value) for value in seed_stability["training_seeds"]
    )
    if (
        len(supplementary_training_seeds) != 3
        or len(set(supplementary_training_seeds)) != 3
        or training_seed not in supplementary_training_seeds
    ):
        raise ValueError(
            "The supplementary analysis requires three unique seeds including the main seed"
        )
    protocol = ProtoNetTrainingProtocol(
        learning_rate_candidates=candidates,
        adam_betas=(betas[0], betas[1]),
        adam_epsilon=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        learning_rate_selection_split=str(selection["split_id"]),
        lr_schedule=payload.get("learning_rate_schedule", "constant"),
        mixed_precision_on_cuda=bool(payload["mixed_precision_on_cuda"]),
        resource_monitor_interval_episodes=int(
            payload["resource_monitor_interval_episodes"]
        ),
        training_seed=training_seed,
        supplementary_seed_split=str(seed_stability["split_id"]),
        supplementary_training_seeds=supplementary_training_seeds,
        validation_interval_episodes=episode_protocol.validation_interval_episodes,
        minimum_train_episodes=episode_protocol.minimum_train_episodes,
        early_stopping_patience_checks=episode_protocol.early_stopping_patience_checks,
        maximum_train_episodes=episode_protocol.maximum_train_episodes,
    )
    if protocol.resource_monitor_interval_episodes < 1:
        raise ValueError("Resource monitor interval must be positive")
    return protocol


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_training_seed(
    protocol: ProtoNetTrainingProtocol,
    split_id: str,
    requested_seed: int | None,
) -> int:
    """Allow the main seed everywhere and supplementary seeds on split 01 only."""

    selected_seed = protocol.training_seed if requested_seed is None else int(requested_seed)
    return selected_seed


def episode_macro_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    way: int,
) -> float:
    scores = []
    for class_index in range(way):
        predicted_class = predictions == class_index
        target_class = targets == class_index
        true_positive = int((predicted_class & target_class).sum())
        false_positive = int((predicted_class & ~target_class).sum())
        false_negative = int((~predicted_class & target_class).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _move_episode_to_device(
    episode: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    copied = dict(episode)
    for key in ("support_images", "support_labels", "query_images", "query_labels"):
        copied[key] = episode[key].to(device, non_blocking=device.type == "cuda")
    return copied


def _autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _selection_key(result: ValidationResult) -> tuple[float, float, float]:
    return result.mean_accuracy, result.mean_macro_f1, -result.mean_loss


def evaluate_protonet(
    model: torch.nn.Module,
    episodes: Iterable[dict[str, object]],
    device: torch.device,
    *,
    mixed_precision: bool,
) -> ValidationResult:
    """Evaluate every fixed validation episode and retain per-query outputs."""

    model.eval()
    accuracies: list[float] = []
    macro_f1_scores: list[float] = []
    losses: list[float] = []
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for raw_episode in episodes:
            episode = _move_episode_to_device(raw_episode, device)
            with _autocast_context(device, mixed_precision):
                output = model(
                    episode["support_images"],
                    episode["support_labels"],
                    episode["query_images"],
                    episode["query_labels"],
                    episode_class_ids=tuple(episode["class_ids"]),
                )
            predictions = output.predictions.detach().cpu()
            targets = episode["query_labels"].detach().cpu()
            probabilities = output.probabilities.detach().float().cpu()
            distances = output.squared_distances.detach().float().cpu()
            relation_scores = (
                output.relation_scores.detach().float().cpu()
                if hasattr(output, "relation_scores")
                else None
            )
            per_query_loss = output.per_query_loss.detach().float().cpu()
            accuracies.append(float((predictions == targets).float().mean()))
            macro_f1_scores.append(episode_macro_f1(predictions, targets, model.way))
            losses.append(float(output.loss.detach().float().cpu()))
            for query_index, metadata in enumerate(raw_episode["query_metadata"]):
                row = {
                    "episode_id": raw_episode["episode_id"],
                    "split_id": raw_episode["split_id"],
                    "shot": raw_episode["shot"],
                    "query_index": query_index,
                    "class_id": metadata["class_id"],
                    "patient_id": metadata["patient_id"],
                    "image_id": metadata["image_id"],
                    "source_relpath": metadata["source_relpath"],
                    "target_label": int(targets[query_index]),
                    "predicted_label": int(predictions[query_index]),
                    "correct": bool(predictions[query_index] == targets[query_index]),
                    "confidence": float(probabilities[query_index].max()),
                    "query_loss": float(per_query_loss[query_index]),
                    "probabilities": probabilities[query_index].tolist(),
                    "squared_distances": distances[query_index].tolist(),
                }
                if relation_scores is not None:
                    row["relation_scores"] = relation_scores[query_index].tolist()
                rows.append(row)
    if not accuracies:
        raise ValueError("Validation loader produced no episodes")
    return ValidationResult(
        mean_accuracy=sum(accuracies) / len(accuracies),
        mean_macro_f1=sum(macro_f1_scores) / len(macro_f1_scores),
        mean_loss=sum(losses) / len(losses),
        episode_count=len(accuracies),
        query_count=len(rows),
        prediction_rows=tuple(rows),
    )


def _write_jsonl_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=raw_handle,
            mtime=0,
        ) as compressed_handle:
            with io.TextIOWrapper(compressed_handle, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    handle.write("\n")


def _append_csv_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    completed_episode: int,
    best_key: tuple[float, float, float] | None,
    best_validation: dict[str, object] | None,
    checks_without_improvement: int,
    learning_rate: float,
    shot: int,
    split_id: str,
    training_seed: int,
    model_id: str,
    model_name: str,
    experiment_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_name": model_name,
        "experiment_metadata": dict(experiment_metadata or {}),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "completed_episode": completed_episode,
        "best_key": best_key,
        "best_validation": best_validation,
        "checks_without_improvement": checks_without_improvement,
        "learning_rate": learning_rate,
        "shot": shot,
        "split_id": split_id,
        "training_seed": training_seed,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _inference_checkpoint_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    """Drop resume-only state from a validation-selected model artifact."""

    retained = (
        "model_state",
        "completed_episode",
        "best_key",
        "best_validation",
        "learning_rate",
        "shot",
        "split_id",
        "training_seed",
        "model_id",
        "model_name",
        "experiment_metadata",
    )
    return {
        "checkpoint_purpose": "inference_best",
        **{key: payload[key] for key in retained},
    }


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    expected_learning_rate: float,
    expected_shot: int,
    expected_split_id: str,
    expected_training_seed: int,
    expected_model_id: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_purpose") == "inference_best":
        raise ValueError(
            "The best checkpoint is inference-only; resume from latest_checkpoint.pt"
        )
    if payload.get("model_id", expected_model_id) != expected_model_id:
        raise ValueError("Resume checkpoint belongs to a different model")
    if not math.isclose(float(payload["learning_rate"]), expected_learning_rate):
        raise ValueError("Resume checkpoint uses a different learning rate")
    if int(payload["shot"]) != expected_shot or payload["split_id"] != expected_split_id:
        raise ValueError("Resume checkpoint belongs to a different experiment")
    if int(payload.get("training_seed", -1)) != expected_training_seed:
        raise ValueError("Resume checkpoint uses a different training seed")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and payload["cuda_random_states"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_states"])
    return payload


def collect_environment(device: torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def train_protonet_run(
    model: torch.nn.Module,
    train_episodes: Iterable[dict[str, object]],
    validation_episodes: Iterable[dict[str, object]],
    protocol: ProtoNetTrainingProtocol,
    *,
    learning_rate: float,
    shot: int,
    split_id: str,
    device: torch.device,
    output_dir: Path,
    resume_checkpoint: Path | None = None,
    training_seed: int | None = None,
    model_id: str = "M0",
    model_name: str = "ProtoNet",
    experiment_metadata: Mapping[str, object] | None = None,
    weight_decay_override: float | None = None,
    attention_lr_mult: float = 1.0,
    backbone_lr_mult: float = 1.0,
    lr_schedule_override: str | None = None,
    skip_lr_check: bool = False,
) -> dict[str, object]:
    """Train one episodic split/shot/learning-rate run with frozen stopping rules."""

    if not skip_lr_check and learning_rate not in protocol.learning_rate_candidates:
        raise ValueError("Learning rate is outside the approved candidate set")
    selected_training_seed = resolve_training_seed(
        protocol,
        split_id,
        training_seed,
    )
    set_reproducible_seed(selected_training_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    _wd = weight_decay_override if weight_decay_override is not None else protocol.weight_decay
    # Build parameter groups: backbone (layer3/4), attention, and other (fc/head)
    _has_discriminative = (attention_lr_mult != 1.0) or (backbone_lr_mult != 1.0)
    if _has_discriminative:
        backbone_params = []
        attn_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "attention" in name or "attn" in name or "cbam" in name:
                attn_params.append(param)
            elif "layer3" in name or "layer4" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        param_groups = [
            {"params": head_params, "lr": learning_rate, "weight_decay": _wd},
            {"params": backbone_params, "lr": learning_rate * backbone_lr_mult, "weight_decay": _wd},
            {"params": attn_params, "lr": learning_rate * attention_lr_mult, "weight_decay": _wd},
        ]
        optimizer = torch.optim.Adam(param_groups, betas=protocol.adam_betas, eps=protocol.adam_epsilon)
        print(f"[Optimizer] head={len(head_params)}(lr={learning_rate}), backbone={len(backbone_params)}(lr={learning_rate * backbone_lr_mult}), attn={len(attn_params)}(lr={learning_rate * attention_lr_mult}), wd={_wd}", flush=True)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=protocol.adam_betas,
            eps=protocol.adam_epsilon,
            weight_decay=_wd,
        )
        print(f"[Optimizer] all params lr={learning_rate}, wd={_wd}", flush=True)
    # Cosine annealing LR scheduler
    lr_scheduler = None
    _lr_schedule = lr_schedule_override if lr_schedule_override is not None else getattr(protocol, "lr_schedule", "constant")
    if _lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=protocol.maximum_train_episodes, eta_min=learning_rate * 0.01
        )
    mixed_precision = protocol.mixed_precision_on_cuda and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    completed_episode = 0
    best_key: tuple[float, float, float] | None = None
    best_validation: dict[str, object] | None = None
    checks_without_improvement = 0
    if resume_checkpoint is not None:
        state = _load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scaler,
            expected_learning_rate=learning_rate,
            expected_shot=shot,
            expected_split_id=split_id,
            expected_training_seed=selected_training_seed,
            expected_model_id=model_id,
        )
        completed_episode = int(state["completed_episode"])
        best_key = tuple(state["best_key"]) if state["best_key"] is not None else None
        best_validation = state["best_validation"]
        checks_without_improvement = int(state["checks_without_improvement"])

    (output_dir / "environment.json").write_text(
        json.dumps(collect_environment(device), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_config = {
        "model_id": model_id,
        "model_name": model_name,
        "split_id": split_id,
        "shot": shot,
        "learning_rate": learning_rate,
        "training_seed": selected_training_seed,
        "training_protocol": asdict(protocol),
        "mixed_precision_enabled": mixed_precision,
        "experiment_metadata": dict(experiment_metadata or {}),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stopped_early = False
    final_episode = completed_episode
    pending_training_rows: list[dict[str, object]] = []
    for raw_episode in train_episodes:
        episode_index = int(str(raw_episode["episode_id"]).rsplit(":", 1)[-1])
        if episode_index <= completed_episode:
            continue
        if episode_index > protocol.maximum_train_episodes:
            break
        final_episode = episode_index
        model.train()
        episode = _move_episode_to_device(raw_episode, device)
        optimizer.zero_grad(set_to_none=True)
        resource_sampled = (
            device.type == "cuda"
            and episode_index % protocol.resource_monitor_interval_episodes == 0
        )
        if resource_sampled:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with _autocast_context(device, mixed_precision):
            output = model(
                episode["support_images"],
                episode["support_labels"],
                episode["query_images"],
                episode["query_labels"],
                episode_class_ids=tuple(episode["class_ids"]),
            )
        scaler.scale(output.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if resource_sampled:
            torch.cuda.synchronize(device)
        duration = (
            time.perf_counter() - started
            if device.type != "cuda" or resource_sampled
            else None
        )
        accuracy = float(
            (output.predictions.detach() == episode["query_labels"]).float().mean()
        )
        pending_training_rows.append(
            {
                "episode_index": episode_index,
                "episode_id": raw_episode["episode_id"],
                "loss": float(output.loss.detach().float().cpu()),
                "accuracy": accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "resource_sampled": resource_sampled,
                "duration_seconds": duration,
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if resource_sampled
                    else None
                ),
            }
        )

        if episode_index % protocol.validation_interval_episodes != 0:
            continue
        _append_csv_rows(
            output_dir / "training_history.csv",
            pending_training_rows,
        )
        pending_training_rows.clear()
        validation = evaluate_protonet(
            model,
            validation_episodes,
            device,
            mixed_precision=mixed_precision,
        )
        current_key = _selection_key(validation)
        improved = best_key is None or current_key > best_key
        validation_row = {
            "train_episode_index": episode_index,
            "mean_accuracy": validation.mean_accuracy,
            "mean_macro_f1": validation.mean_macro_f1,
            "mean_loss": validation.mean_loss,
            "validation_episode_count": validation.episode_count,
            "query_count": validation.query_count,
            "improved": improved,
        }
        _append_csv_rows(output_dir / "validation_history.csv", [validation_row])
        if improved:
            best_key = current_key
            checks_without_improvement = 0
            best_validation = validation_row
            _write_jsonl_gz(
                output_dir / "best_validation_predictions.jsonl.gz",
                validation.prediction_rows,
            )
        else:
            checks_without_improvement += 1

        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scaler,
            completed_episode=episode_index,
            best_key=best_key,
            best_validation=best_validation,
            checks_without_improvement=checks_without_improvement,
            learning_rate=learning_rate,
            shot=shot,
            split_id=split_id,
            training_seed=selected_training_seed,
            model_id=model_id,
            model_name=model_name,
            experiment_metadata=experiment_metadata,
        )
        _save_checkpoint(output_dir / "latest_checkpoint.pt", checkpoint)
        if improved:
            _save_checkpoint(
                output_dir / "best_checkpoint.pt",
                _inference_checkpoint_payload(checkpoint),
            )
        if (
            episode_index >= protocol.minimum_train_episodes
            and checks_without_improvement
            >= protocol.early_stopping_patience_checks
        ):
            stopped_early = True
            break

    _append_csv_rows(output_dir / "training_history.csv", pending_training_rows)
    if final_episode == completed_episode:
        raise RuntimeError("Training loader produced no new episodes")
    summary = {
        "model_id": model_id,
        "model_name": model_name,
        "split_id": split_id,
        "shot": shot,
        "learning_rate": learning_rate,
        "training_seed": selected_training_seed,
        "completed_episode": final_episode,
        "stopped_early": stopped_early,
        "best_validation": best_validation,
        "best_selection_key": best_key,
        "experiment_metadata": dict(experiment_metadata or {}),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


# Public neutral name for M3 and later episodic models. The historical M0 name
# remains available so existing commands and checkpoints do not break.
train_episodic_run = train_protonet_run


def select_common_learning_rate(
    summaries: Sequence[dict[str, object]],
    candidates: Sequence[float] = (0.0001, 0.0003),
    selection_split_id: str = "split_01",
    primary_training_seed: int = 20261004,
) -> dict[str, object]:
    """Select one learning rate by averaging split-01 validation across both shots."""

    by_rate: dict[float, list[dict[str, object]]] = {
        float(rate): [] for rate in candidates
    }
    for summary in summaries:
        if summary.get("split_id") != selection_split_id:
            raise ValueError("Learning-rate selection may only use split_01 validation")
        if int(summary.get("training_seed", -1)) != primary_training_seed:
            raise ValueError("Learning-rate selection may only use the primary training seed")
        rate = float(summary["learning_rate"])
        if rate not in by_rate:
            raise ValueError(f"Unexpected learning rate in summary: {rate}")
        by_rate[rate].append(summary)
    ranking = []
    for rate, members in by_rate.items():
        shots = {int(member["shot"]) for member in members}
        if shots != {3, 5} or len(members) != 2:
            raise ValueError(f"Learning rate {rate} must have both 3-shot and 5-shot runs")
        validations = [member["best_validation"] for member in members]
        if any(item is None for item in validations):
            raise ValueError(f"Learning rate {rate} has no completed validation result")
        mean_accuracy = sum(float(item["mean_accuracy"]) for item in validations) / len(validations)
        mean_macro_f1 = sum(float(item["mean_macro_f1"]) for item in validations) / len(validations)
        mean_loss = sum(float(item["mean_loss"]) for item in validations) / len(validations)
        ranking.append(
            {
                "learning_rate": rate,
                "mean_validation_accuracy": mean_accuracy,
                "mean_validation_macro_f1": mean_macro_f1,
                "mean_validation_loss": mean_loss,
                "selection_key": (mean_accuracy, mean_macro_f1, -mean_loss),
            }
        )
    winner = max(ranking, key=lambda item: item["selection_key"])
    return {
        "selection_split_id": selection_split_id,
        "training_seed": primary_training_seed,
        "selected_learning_rate": winner["learning_rate"],
        "ranking": sorted(
            ranking,
            key=lambda item: item["selection_key"],
            reverse=True,
        ),
    }
