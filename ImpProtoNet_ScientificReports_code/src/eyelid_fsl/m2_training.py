"""Supervised encoder training, selection, and checkpointing for M2."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch.nn import functional

from .data.episodes import ClassPool
from .data.supervised_dataset import (
    SupervisedPatientDataset,
    build_supervised_dataloader,
    build_supervised_epoch_samples,
)
from .data.imaging import deterministic_horizontal_flip
from .supervised_transfer import (
    M2ValidationResult,
    SupervisedTransferModel,
    SupervisedTransferProtocol,
    evaluate_m2_validation,
    resolve_m2_training_seed,
    select_best_m2_validation,
)
from .training import collect_environment, set_reproducible_seed


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


def _write_jsonl_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw_handle, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    )
                    handle.write("\n")


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _inference_encoder_checkpoint_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    """Keep only the selected encoder and provenance needed at M2 test time."""

    retained = (
        "model_id",
        "encoder_state",
        "completed_epoch",
        "best_key",
        "best_validation",
        "learning_rate",
        "split_id",
        "training_seed",
    )
    return {
        "checkpoint_purpose": "inference_best_encoder",
        **{key: payload[key] for key in retained},
    }


def _validation_row(
    result: M2ValidationResult,
    *,
    epoch: int,
    improved: bool,
    selected_at_check: bool,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "regularization_c": result.regularization_c,
        "mean_accuracy": result.mean_accuracy,
        "mean_macro_f1": result.mean_macro_f1,
        "mean_loss": result.mean_loss,
        "validation_episode_count": result.episode_count,
        "query_count": result.query_count,
        "shot_3_accuracy": result.shot_metrics[3]["mean_accuracy"],
        "shot_3_macro_f1": result.shot_metrics[3]["mean_macro_f1"],
        "shot_3_loss": result.shot_metrics[3]["mean_loss"],
        "shot_5_accuracy": result.shot_metrics[5]["mean_accuracy"],
        "shot_5_macro_f1": result.shot_metrics[5]["mean_macro_f1"],
        "shot_5_loss": result.shot_metrics[5]["mean_loss"],
        "selected_at_check": selected_at_check,
        "improved_global_best": improved,
    }


def _checkpoint_payload(
    model: SupervisedTransferModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    completed_epoch: int,
    best_key: tuple[float, float, float] | None,
    best_validation: dict[str, object] | None,
    checks_without_improvement: int,
    learning_rate: float,
    split_id: str,
    training_seed: int,
) -> dict[str, object]:
    return {
        "model_id": "M2",
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "number_of_classes": model.number_of_classes,
        "completed_epoch": completed_epoch,
        "best_key": best_key,
        "best_validation": best_validation,
        "checks_without_improvement": checks_without_improvement,
        "learning_rate": learning_rate,
        "split_id": split_id,
        "training_seed": training_seed,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _load_resume_checkpoint(
    path: Path,
    model: SupervisedTransferModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    expected_learning_rate: float,
    expected_split_id: str,
    expected_training_seed: int,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_purpose") == "inference_best_encoder":
        raise ValueError(
            "The M2 best checkpoint is inference-only; resume from latest_checkpoint.pt"
        )
    if payload.get("model_id") != "M2":
        raise ValueError("Resume checkpoint is not an M2 checkpoint")
    if payload["split_id"] != expected_split_id:
        raise ValueError("Resume checkpoint belongs to a different split")
    if int(payload["training_seed"]) != expected_training_seed:
        raise ValueError("Resume checkpoint uses a different training seed")
    if not math.isclose(float(payload["learning_rate"]), expected_learning_rate):
        raise ValueError("Resume checkpoint uses a different learning rate")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and payload["cuda_random_states"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_states"])
    return payload


def _validation_c_values(
    protocol: SupervisedTransferProtocol,
    *,
    split_id: str,
    training_seed: int,
    selected_regularization_c: float | None,
) -> tuple[float, ...]:
    if selected_regularization_c is not None:
        selected = float(selected_regularization_c)
        if selected not in protocol.regularization_c_candidates:
            raise ValueError("selected_regularization_c is outside the approved set")
        return (selected,)
    if split_id != protocol.selection_split_id or training_seed != protocol.training_seed:
        raise ValueError(
            "A fixed selected_regularization_c is required outside the split_01 primary run"
        )
    return protocol.regularization_c_candidates


def train_supervised_transfer_run(
    model: SupervisedTransferModel,
    class_pools: Mapping[str, ClassPool],
    source_root: Path,
    validation_episodes_by_shot: Mapping[int, Iterable[dict[str, object]]],
    protocol: SupervisedTransferProtocol,
    *,
    learning_rate: float,
    split_id: str,
    device: torch.device,
    output_dir: Path,
    selected_regularization_c: float | None = None,
    resume_checkpoint: Path | None = None,
    training_seed: int | None = None,
    num_workers: int = 4,
    image_size: int = 224,
) -> dict[str, object]:
    """Train one encoder shared by the 3-shot and 5-shot M2 evaluations."""

    if learning_rate not in protocol.learning_rate_candidates:
        raise ValueError("Learning rate is outside the approved M2 candidate set")
    seed = resolve_m2_training_seed(protocol, split_id, training_seed)
    c_values = _validation_c_values(
        protocol,
        split_id=split_id,
        training_seed=seed,
        selected_regularization_c=selected_regularization_c,
    )
    if len(class_pools) != model.number_of_classes:
        raise ValueError("Model class count does not match the split training classes")
    set_reproducible_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=protocol.adam_betas,
        eps=protocol.adam_epsilon,
        weight_decay=protocol.weight_decay,
    )
    mixed_precision = protocol.mixed_precision_on_cuda and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    completed_epoch = 0
    best_key: tuple[float, float, float] | None = None
    best_validation: dict[str, object] | None = None
    checks_without_improvement = 0
    if resume_checkpoint is not None:
        state = _load_resume_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scaler,
            expected_learning_rate=learning_rate,
            expected_split_id=split_id,
            expected_training_seed=seed,
        )
        completed_epoch = int(state["completed_epoch"])
        best_key = tuple(state["best_key"]) if state["best_key"] else None
        best_validation = state["best_validation"]
        checks_without_improvement = int(state["checks_without_improvement"])

    (output_dir / "environment.json").write_text(
        json.dumps(collect_environment(device), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "split_id": split_id,
                "learning_rate": learning_rate,
                "training_seed": seed,
                "validation_regularization_c_values": c_values,
                "training_protocol": asdict(protocol),
                "training_class_ids": sorted(class_pools),
                "unique_training_patients": sum(
                    len(pool.images_by_patient) for pool in class_pools.values()
                ),
                "mixed_precision_enabled": mixed_precision,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stopped_early = False
    final_epoch = completed_epoch
    for epoch in range(completed_epoch + 1, protocol.maximum_epochs + 1):
        final_epoch = epoch
        samples = build_supervised_epoch_samples(
            class_pools, epoch=epoch, master_seed=seed
        )
        sample_rows = [
            {
                "epoch": sample.epoch,
                "sample_index": sample.sample_index,
                "class_id": sample.class_id,
                "class_label": sample.class_label,
                "patient_id": sample.patient_id,
                "image_id": sample.image_id,
                "source_relpath": sample.source_relpath,
                "horizontal_flipped": deterministic_horizontal_flip(
                    0.5,
                    seed,
                    "supervised",
                    sample.epoch,
                    sample.sample_index,
                    sample.image_id,
                ),
            }
            for sample in samples
        ]
        _append_csv_rows(output_dir / "training_sample_history.csv", sample_rows)
        dataset = SupervisedPatientDataset(
            samples,
            source_root,
            augmentation_seed=seed,
            image_size=image_size,
        )
        loader = build_supervised_dataloader(
            dataset,
            batch_size=protocol.batch_size,
            device=device,
            num_workers=num_workers,
            seed=seed + epoch,
        )
        model.train()
        resource_sampled = (
            device.type == "cuda"
            and epoch % protocol.resource_monitor_interval_epochs == 0
        )
        if resource_sampled:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_losses: list[float] = []
        epoch_correct = 0
        epoch_count = 0
        batch_rows: list[dict[str, object]] = []
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with torch.amp.autocast(
                device_type=device.type, enabled=mixed_precision
            ):
                logits = model(images)
                loss = functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            duration = (
                time.perf_counter() - started if device.type != "cuda" else None
            )
            predictions = logits.detach().argmax(dim=1)
            correct = int((predictions == labels).sum())
            count = int(labels.numel())
            epoch_losses.append(float(loss.detach().float().cpu()) * count)
            epoch_correct += correct
            epoch_count += count
            batch_rows.append(
                {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "sample_count": count,
                    "loss": float(loss.detach().float().cpu()),
                    "accuracy": correct / count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "duration_seconds": duration,
                    "peak_gpu_memory_bytes": None,
                }
            )
        if epoch_count == 0:
            raise RuntimeError("Supervised training loader produced no samples")
        if resource_sampled:
            torch.cuda.synchronize(device)
        epoch_duration = time.perf_counter() - epoch_started
        _append_csv_rows(output_dir / "training_batch_history.csv", batch_rows)
        _append_csv_rows(
            output_dir / "training_epoch_history.csv",
            [
                {
                    "epoch": epoch,
                    "sample_count": epoch_count,
                    "mean_loss": sum(epoch_losses) / epoch_count,
                    "accuracy": epoch_correct / epoch_count,
                    "resource_sampled": resource_sampled,
                    "duration_seconds": epoch_duration,
                    "peak_gpu_memory_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if resource_sampled
                        else None
                    ),
                }
            ],
        )
        if epoch % protocol.validation_interval_epochs != 0:
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                scaler,
                completed_epoch=epoch,
                best_key=best_key,
                best_validation=best_validation,
                checks_without_improvement=checks_without_improvement,
                learning_rate=learning_rate,
                split_id=split_id,
                training_seed=seed,
            )
            _save_checkpoint(output_dir / "latest_checkpoint.pt", checkpoint)
            continue

        validation_results = evaluate_m2_validation(
            model.encoder,
            validation_episodes_by_shot,
            c_values,
            device=device,
            mixed_precision=mixed_precision,
            maximum_iterations=protocol.logistic_maximum_iterations,
            tolerance=protocol.logistic_tolerance,
        )
        selected = select_best_m2_validation(validation_results)
        current_key = selected.selection_key
        improved = best_key is None or current_key > best_key
        validation_rows = [
            _validation_row(
                result,
                epoch=epoch,
                selected_at_check=result is selected,
                improved=improved and result is selected,
            )
            for result in validation_results
        ]
        _append_csv_rows(output_dir / "validation_history.csv", validation_rows)
        if improved:
            best_key = current_key
            checks_without_improvement = 0
            best_validation = next(
                row for row in validation_rows if row["selected_at_check"]
            )
            _write_jsonl_gz(
                output_dir / "best_validation_predictions.jsonl.gz",
                selected.prediction_rows,
            )
        else:
            checks_without_improvement += 1
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scaler,
            completed_epoch=epoch,
            best_key=best_key,
            best_validation=best_validation,
            checks_without_improvement=checks_without_improvement,
            learning_rate=learning_rate,
            split_id=split_id,
            training_seed=seed,
        )
        _save_checkpoint(output_dir / "latest_checkpoint.pt", checkpoint)
        if improved:
            _save_checkpoint(
                output_dir / "best_checkpoint.pt",
                _inference_encoder_checkpoint_payload(checkpoint),
            )
        if (
            epoch >= protocol.minimum_epochs
            and checks_without_improvement
            >= protocol.early_stopping_patience_checks
        ):
            stopped_early = True
            break

    if best_validation is None:
        raise RuntimeError("Training completed without an M2 validation check")
    summary = {
        "model_id": "M2",
        "split_id": split_id,
        "learning_rate": learning_rate,
        "training_seed": seed,
        "completed_epoch": final_epoch,
        "stopped_early": stopped_early,
        "best_validation": best_validation,
        "best_selection_key": best_key,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def select_m2_hyperparameters(
    summaries: Sequence[dict[str, object]],
    *,
    candidates: Sequence[float] = (0.0001, 0.0003),
    selection_split_id: str = "split_01",
    primary_training_seed: int = 20261004,
) -> dict[str, object]:
    """Choose one LR/C pair from the two split-01 supervised encoder runs."""

    expected_rates = {float(rate) for rate in candidates}
    if len(summaries) != len(expected_rates):
        raise ValueError("Exactly one summary is required for each learning rate")
    ranking: list[dict[str, object]] = []
    observed_rates: set[float] = set()
    for summary in summaries:
        if summary.get("model_id") != "M2":
            raise ValueError("Hyperparameter selection only accepts M2 summaries")
        if summary.get("split_id") != selection_split_id:
            raise ValueError("M2 selection may only use split_01 validation")
        if int(summary.get("training_seed", -1)) != primary_training_seed:
            raise ValueError("M2 selection may only use the primary training seed")
        rate = float(summary["learning_rate"])
        if rate not in expected_rates or rate in observed_rates:
            raise ValueError("Unexpected or duplicate M2 learning rate")
        observed_rates.add(rate)
        validation = summary.get("best_validation")
        if not validation:
            raise ValueError("M2 summary has no completed validation")
        key = (
            float(validation["mean_accuracy"]),
            float(validation["mean_macro_f1"]),
            -float(validation["mean_loss"]),
        )
        ranking.append(
            {
                "learning_rate": rate,
                "regularization_c": float(validation["regularization_c"]),
                "mean_validation_accuracy": key[0],
                "mean_validation_macro_f1": key[1],
                "mean_validation_loss": -key[2],
                "selection_key": key,
            }
        )
    if observed_rates != expected_rates:
        raise ValueError("M2 learning-rate candidate coverage is incomplete")
    ranking.sort(key=lambda item: item["selection_key"], reverse=True)
    winner = ranking[0]
    return {
        "selection_split_id": selection_split_id,
        "training_seed": primary_training_seed,
        "selected_learning_rate": winner["learning_rate"],
        "selected_regularization_c": winner["regularization_c"],
        "ranking": ranking,
    }


def load_best_m2_encoder_checkpoint(
    encoder: torch.nn.Module,
    path: Path,
    *,
    expected_split_id: str,
    expected_training_seed: int,
    expected_learning_rate: float | None = None,
    expected_regularization_c: float | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_id") != "M2":
        raise ValueError("Checkpoint is not an M2 checkpoint")
    if payload.get("split_id") != expected_split_id:
        raise ValueError("M2 checkpoint belongs to a different split")
    if expected_training_seed is not None and int(payload.get("training_seed", -1)) != expected_training_seed:
        raise ValueError("M2 checkpoint uses a different training seed")
    if expected_learning_rate is not None and not math.isclose(
        float(payload["learning_rate"]), expected_learning_rate
    ):
        raise ValueError("M2 checkpoint uses a different learning rate")
    if payload.get("best_validation") is None:
        raise ValueError("M2 checkpoint has no validation-selected state")
    if expected_regularization_c is not None and not math.isclose(
        float(payload["best_validation"]["regularization_c"]),
        expected_regularization_c,
    ):
        raise ValueError("M2 checkpoint uses a different logistic-regression C")
    encoder.load_state_dict(payload["encoder_state"])
    return payload
