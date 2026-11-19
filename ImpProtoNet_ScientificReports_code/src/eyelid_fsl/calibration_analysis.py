"""Experiment C uncalibrated confidence, calibration, and selective prediction."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class CalibrationProtocol:
    shot: int
    expected_episodes_per_run: int
    expected_queries_per_episode: int
    model_ids: tuple[str, ...]
    primary_model_ids: tuple[str, ...]
    supplementary_model_ids: tuple[str, ...]
    probability_clip_epsilon: float
    ece_bin_count: int
    high_confidence_thresholds: tuple[float, ...]
    risk_coverage_levels: tuple[float, ...]
    bootstrap_replicates: int
    confidence_level: float
    bootstrap_seed: int


@dataclass(frozen=True)
class Prediction:
    model_id: str
    split_id: str
    episode_id: str
    patient_id: str
    image_id: str
    target_class_id: str
    predicted_class_id: str
    correct: bool
    confidence: float
    true_class_probability: float
    probabilities: tuple[float, ...]
    class_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def load_calibration_protocol(path: Path) -> CalibrationProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "experiment_id": "C",
        "shot": 5,
        "expected_episodes_per_run": 1000,
        "expected_queries_per_episode": 15,
        "model_ids": ["M0", "M1", "M2", "M3", "M5"],
        "primary_model_ids": ["M0", "M1", "M2", "M3"],
        "supplementary_model_ids": ["M5"],
        "excluded_model_ids": ["M4"],
        "probability_postprocessing": "none",
        "probability_clip_epsilon": 1e-15,
        "ece": {
            "binning": "equal_width",
            "bin_count": 10,
            "interval": "left_closed_right_open_except_final",
        },
        "high_confidence_thresholds": [0.8, 0.9],
        "risk_coverage_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "bootstrap": {
            "replicates": 10000,
            "confidence_level": 0.95,
            "seed": 20261004,
            "cluster": ["class_split", "disease_class", "patient"],
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Experiment C calibration configuration changed: {key}")
    return CalibrationProtocol(
        shot=5,
        expected_episodes_per_run=1000,
        expected_queries_per_episode=15,
        model_ids=tuple(expected["model_ids"]),
        primary_model_ids=tuple(expected["primary_model_ids"]),
        supplementary_model_ids=("M5",),
        probability_clip_epsilon=1e-15,
        ece_bin_count=10,
        high_confidence_thresholds=(0.8, 0.9),
        risk_coverage_levels=tuple(expected["risk_coverage_levels"]),
        bootstrap_replicates=10000,
        confidence_level=0.95,
        bootstrap_seed=20261004,
    )


def _load_predictions(
    path: Path,
    *,
    model_id: str,
    split_id: str,
    protocol: CalibrationProtocol,
) -> list[Prediction]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    episode_counts: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model_id") != model_id or row.get("split_id") != split_id:
                raise ValueError("Calibration input identity does not match its directory")
            if int(row.get("shot", -1)) != protocol.shot or row.get("partition") != "test":
                raise ValueError("Experiment C accepts 5-shot test predictions only")
            probability_map = row.get("probabilities_by_class")
            if not isinstance(probability_map, dict) or len(probability_map) != 3:
                raise ValueError("Calibration requires a complete three-class probability vector")
            class_ids = tuple(str(value) for value in probability_map)
            probabilities = tuple(float(probability_map[value]) for value in probability_map)
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
                raise ValueError("Predicted probabilities must be finite and in [0, 1]")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
                raise ValueError("Predicted probability vector does not sum to one")
            target = str(row["target_class_id"])
            predicted = str(row["predicted_class_id"])
            if target not in probability_map or predicted not in probability_map:
                raise ValueError("Target or predicted class is absent from probability vector")
            confidence = float(row["confidence"])
            true_probability = float(row["true_class_probability"])
            if not math.isclose(confidence, max(probabilities), abs_tol=1e-6):
                raise ValueError("Stored confidence disagrees with probability vector")
            if not math.isclose(true_probability, float(probability_map[target]), abs_tol=1e-6):
                raise ValueError("Stored true-class probability disagrees with vector")
            correct = bool(row["correct"])
            if correct != (target == predicted):
                raise ValueError("Stored correctness disagrees with labels")
            episode_id = str(row["episode_id"])
            episode_counts[episode_id] = episode_counts.get(episode_id, 0) + 1
            rows.append(
                Prediction(
                    model_id=model_id,
                    split_id=split_id,
                    episode_id=episode_id,
                    patient_id=str(row["patient_id"]),
                    image_id=str(row["image_id"]),
                    target_class_id=target,
                    predicted_class_id=predicted,
                    correct=correct,
                    confidence=confidence,
                    true_class_probability=true_probability,
                    probabilities=probabilities,
                    class_ids=class_ids,
                )
            )
    if len(episode_counts) != protocol.expected_episodes_per_run:
        raise ValueError("Calibration run has an unexpected episode count")
    if any(count != protocol.expected_queries_per_episode for count in episode_counts.values()):
        raise ValueError("Calibration episode has an incomplete query set")
    return rows


def _bin_index(confidence: float, bin_count: int) -> int:
    return min(bin_count - 1, int(confidence * bin_count))


def _brier(prediction: Prediction) -> float:
    return sum(
        (probability - float(class_id == prediction.target_class_id)) ** 2
        for class_id, probability in zip(prediction.class_ids, prediction.probabilities)
    )


def _calibration_metrics(
    predictions: Sequence[Prediction], protocol: CalibrationProtocol
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if not predictions:
        raise ValueError("Calibration metrics require predictions")
    bin_values: list[list[Prediction]] = [[] for _ in range(protocol.ece_bin_count)]
    for prediction in predictions:
        bin_values[_bin_index(prediction.confidence, protocol.ece_bin_count)].append(prediction)
    bin_rows = []
    ece = 0.0
    for index, selected in enumerate(bin_values):
        count = len(selected)
        mean_confidence = statistics.mean(item.confidence for item in selected) if selected else 0.0
        empirical_accuracy = statistics.mean(item.correct for item in selected) if selected else 0.0
        contribution = count / len(predictions) * abs(empirical_accuracy - mean_confidence)
        ece += contribution
        bin_rows.append(
            {
                "bin_index": index + 1,
                "lower_bound_inclusive": index / protocol.ece_bin_count,
                "upper_bound": (index + 1) / protocol.ece_bin_count,
                "upper_bound_inclusive": index == protocol.ece_bin_count - 1,
                "prediction_count": count,
                "prediction_fraction": count / len(predictions),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_calibration_gap": abs(empirical_accuracy - mean_confidence),
                "ece_contribution": contribution,
            }
        )
    metrics = {
        "prediction_count": float(len(predictions)),
        "accuracy": statistics.mean(item.correct for item in predictions),
        "mean_confidence": statistics.mean(item.confidence for item in predictions),
        "ece": ece,
        "brier_score": statistics.mean(_brier(item) for item in predictions),
        "negative_log_likelihood": statistics.mean(
            -math.log(max(protocol.probability_clip_epsilon, item.true_class_probability))
            for item in predictions
        ),
    }
    return metrics, bin_rows


def _risk_coverage(
    predictions: Sequence[Prediction], coverage_levels: Sequence[float]
) -> tuple[list[dict[str, object]], float]:
    ordered = sorted(
        predictions,
        key=lambda item: (
            -item.confidence,
            item.split_id,
            item.episode_id,
            item.patient_id,
            item.image_id,
        ),
    )
    cumulative_errors = []
    errors = 0
    for item in ordered:
        errors += int(not item.correct)
        cumulative_errors.append(errors)
    rows = []
    for level in coverage_levels:
        retained = max(1, math.ceil(level * len(ordered)))
        risk = cumulative_errors[retained - 1] / retained
        rows.append(
            {
                "requested_coverage": level,
                "retained_prediction_count": retained,
                "actual_coverage": retained / len(ordered),
                "confidence_cutoff": ordered[retained - 1].confidence,
                "accuracy": 1.0 - risk,
                "risk": risk,
            }
        )
    aurc = statistics.mean(
        error_count / rank for rank, error_count in enumerate(cumulative_errors, start=1)
    )
    return rows, aurc


def _bootstrap_weights(
    cluster_keys: Sequence[tuple[str, str, str]],
    split_ids: Sequence[str],
    replicates: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    index_by_stratum: dict[tuple[str, str], list[int]] = {}
    for index, (split_id, class_id, _patient_id) in enumerate(cluster_keys):
        index_by_stratum.setdefault((split_id, class_id), []).append(index)
    classes_by_split = {
        split_id: sorted(
            class_id for (candidate, class_id) in index_by_stratum if candidate == split_id
        )
        for split_id in split_ids
    }
    weights = torch.zeros((replicates, len(cluster_keys)), dtype=torch.float32)
    split_draws = torch.randint(
        0, len(split_ids), (replicates, len(split_ids)), generator=generator
    )
    one = torch.tensor(1.0)
    for position in range(len(split_ids)):
        for split_index, split_id in enumerate(split_ids):
            selected_replicates = torch.nonzero(
                split_draws[:, position] == split_index, as_tuple=False
            ).reshape(-1)
            if selected_replicates.numel() == 0:
                continue
            for class_id in classes_by_split[split_id]:
                global_indices = index_by_stratum[(split_id, class_id)]
                count = len(global_indices)
                draws = torch.randint(
                    0,
                    count,
                    (selected_replicates.numel(), count),
                    generator=generator,
                )
                local_counts = torch.zeros(
                    (selected_replicates.numel(), count), dtype=torch.float32
                )
                local_counts.scatter_add_(
                    1, draws, one.expand_as(draws).to(torch.float32)
                )
                weights[
                    selected_replicates[:, None],
                    torch.tensor(global_indices)[None, :],
                ] += local_counts
    return weights


def analyse_calibration(
    *,
    results_root: Path,
    output_dir: Path,
    protocol: CalibrationProtocol,
    split_ids: Sequence[str] = tuple(f"split_{i:02d}" for i in range(1, 6)),
    protocol_path: Path | None = None,
) -> dict[str, object]:
    predictions_by_run: dict[tuple[str, str], list[Prediction]] = {}
    source_paths = []
    for model_id, split_id in itertools.product(protocol.model_ids, split_ids):
        path = results_root / model_id / split_id / f"{protocol.shot}shot" / "test_query_predictions.jsonl.gz"
        predictions_by_run[(model_id, split_id)] = _load_predictions(
            path, model_id=model_id, split_id=split_id, protocol=protocol
        )
        source_paths.append(path)
    reference_model = protocol.model_ids[0]
    for split_id in split_ids:
        reference = {
            (item.episode_id, item.patient_id, item.image_id, item.target_class_id)
            for item in predictions_by_run[(reference_model, split_id)]
        }
        for model_id in protocol.model_ids[1:]:
            candidate = {
                (item.episode_id, item.patient_id, item.image_id, item.target_class_id)
                for item in predictions_by_run[(model_id, split_id)]
            }
            if candidate != reference:
                raise ValueError("Calibration models do not share the same query predictions")

    split_metric_rows = []
    model_metric_rows = []
    reliability_rows = []
    high_confidence_rows = []
    risk_coverage_rows = []
    pooled_by_model: dict[str, list[Prediction]] = {}
    for model_id in protocol.model_ids:
        pooled = [
            prediction
            for split_id in split_ids
            for prediction in predictions_by_run[(model_id, split_id)]
        ]
        pooled_by_model[model_id] = pooled
        analysis_tier = "primary" if model_id in protocol.primary_model_ids else "supplementary"
        for split_id in split_ids:
            selected = predictions_by_run[(model_id, split_id)]
            metrics, bins = _calibration_metrics(selected, protocol)
            _curve, aurc = _risk_coverage(selected, protocol.risk_coverage_levels)
            split_metric_rows.append(
                {
                    "model_id": model_id,
                    "analysis_tier": analysis_tier,
                    "split_id": split_id,
                    "shot": protocol.shot,
                    **metrics,
                    "aurc": aurc,
                }
            )
            reliability_rows.extend(
                {
                    "model_id": model_id,
                    "analysis_tier": analysis_tier,
                    "scope": "split",
                    "split_id": split_id,
                    **row,
                }
                for row in bins
            )
        metrics, bins = _calibration_metrics(pooled, protocol)
        curve, aurc = _risk_coverage(pooled, protocol.risk_coverage_levels)
        model_metric_rows.append(
            {
                "model_id": model_id,
                "analysis_tier": analysis_tier,
                "split_count": len(split_ids),
                "shot": protocol.shot,
                **metrics,
                "aurc": aurc,
            }
        )
        reliability_rows.extend(
            {
                "model_id": model_id,
                "analysis_tier": analysis_tier,
                "scope": "pooled",
                "split_id": "all",
                **row,
            }
            for row in bins
        )
        for threshold in protocol.high_confidence_thresholds:
            retained = [item for item in pooled if item.confidence >= threshold]
            high_confidence_rows.append(
                {
                    "model_id": model_id,
                    "analysis_tier": analysis_tier,
                    "confidence_threshold_inclusive": threshold,
                    "total_prediction_count": len(pooled),
                    "retained_prediction_count": len(retained),
                    "coverage": len(retained) / len(pooled),
                    "error_count": sum(not item.correct for item in retained),
                    "error_rate_among_retained": (
                        sum(not item.correct for item in retained) / len(retained)
                        if retained else float("nan")
                    ),
                }
            )
        risk_coverage_rows.extend(
            {
                "model_id": model_id,
                "analysis_tier": analysis_tier,
                "scope": "pooled",
                "split_id": "all",
                **row,
            }
            for row in curve
        )

    reference_clusters = sorted(
        {
            (item.split_id, item.target_class_id, item.patient_id)
            for item in pooled_by_model[reference_model]
        }
    )
    weights = _bootstrap_weights(
        reference_clusters, split_ids, protocol.bootstrap_replicates, protocol.bootstrap_seed
    )
    bootstrap_rows = []
    bootstrap_index = []
    bootstrap_tensors = []
    metric_names = ["accuracy", "ece", "brier_score", "negative_log_likelihood"]
    for threshold in protocol.high_confidence_thresholds:
        metric_names.extend(
            (
                f"high_confidence_coverage_at_{threshold:.2f}",
                f"high_confidence_error_rate_at_{threshold:.2f}",
            )
        )
    for model_id in protocol.model_ids:
        grouped: dict[tuple[str, str, str], list[Prediction]] = {}
        for prediction in pooled_by_model[model_id]:
            grouped.setdefault(
                (prediction.split_id, prediction.target_class_id, prediction.patient_id), []
            ).append(prediction)
        if sorted(grouped) != reference_clusters:
            raise ValueError("Models do not share the same patient clusters")
        feature_count = 5 + protocol.ece_bin_count * 3 + len(protocol.high_confidence_thresholds) * 2
        stats = torch.zeros((len(reference_clusters), feature_count), dtype=torch.float64)
        for index, key in enumerate(reference_clusters):
            selected = grouped[key]
            stats[index, 0] = len(selected)
            stats[index, 1] = sum(item.correct for item in selected)
            stats[index, 2] = sum(item.confidence for item in selected)
            stats[index, 3] = sum(_brier(item) for item in selected)
            stats[index, 4] = sum(
                -math.log(max(protocol.probability_clip_epsilon, item.true_class_probability))
                for item in selected
            )
            offset = 5
            for bin_index in range(protocol.ece_bin_count):
                binned = [item for item in selected if _bin_index(item.confidence, protocol.ece_bin_count) == bin_index]
                stats[index, offset] = len(binned)
                stats[index, offset + 1] = sum(item.confidence for item in binned)
                stats[index, offset + 2] = sum(item.correct for item in binned)
                offset += 3
            for threshold in protocol.high_confidence_thresholds:
                retained = [item for item in selected if item.confidence >= threshold]
                stats[index, offset] = len(retained)
                stats[index, offset + 1] = sum(not item.correct for item in retained)
                offset += 2
        totals = weights.to(torch.float64) @ stats
        total_count = totals[:, 0].clamp_min(1)
        replicates_by_metric = {
            "accuracy": totals[:, 1] / total_count,
            "brier_score": totals[:, 3] / total_count,
            "negative_log_likelihood": totals[:, 4] / total_count,
        }
        ece = torch.zeros(protocol.bootstrap_replicates, dtype=torch.float64)
        offset = 5
        for _bin_index_value in range(protocol.ece_bin_count):
            bin_count = totals[:, offset]
            nonempty = bin_count > 0
            mean_confidence = torch.zeros_like(bin_count)
            accuracy = torch.zeros_like(bin_count)
            mean_confidence[nonempty] = totals[nonempty, offset + 1] / bin_count[nonempty]
            accuracy[nonempty] = totals[nonempty, offset + 2] / bin_count[nonempty]
            ece += bin_count / total_count * (accuracy - mean_confidence).abs()
            offset += 3
        replicates_by_metric["ece"] = ece
        for threshold in protocol.high_confidence_thresholds:
            retained = totals[:, offset]
            errors = totals[:, offset + 1]
            replicates_by_metric[
                f"high_confidence_coverage_at_{threshold:.2f}"
            ] = retained / total_count
            values = torch.full_like(retained, float("nan"))
            values[retained > 0] = errors[retained > 0] / retained[retained > 0]
            replicates_by_metric[f"high_confidence_error_rate_at_{threshold:.2f}"] = values
            offset += 2
        point = next(row for row in model_metric_rows if row["model_id"] == model_id)
        for metric in metric_names:
            values = replicates_by_metric[metric]
            finite = values[torch.isfinite(values)]
            alpha = 1.0 - protocol.confidence_level
            if finite.numel():
                quantiles = torch.quantile(
                    finite,
                    torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64),
                )
                standard_error = (
                    float(finite.std(unbiased=True))
                    if finite.numel() > 1
                    else float("nan")
                )
                ci_lower, ci_upper = float(quantiles[0]), float(quantiles[1])
            else:
                standard_error = float("nan")
                ci_lower = float("nan")
                ci_upper = float("nan")
            if metric.startswith("high_confidence"):
                threshold = float(metric.rsplit("_", 1)[1])
                high_row = next(
                    row for row in high_confidence_rows
                    if row["model_id"] == model_id
                    and math.isclose(float(row["confidence_threshold_inclusive"]), threshold)
                )
                point_estimate = float(
                    high_row[
                        "coverage"
                        if "_coverage_" in metric
                        else "error_rate_among_retained"
                    ]
                )
            else:
                point_estimate = float(point[metric])
            index = len(bootstrap_tensors)
            bootstrap_rows.append(
                {
                    "bootstrap_index": index,
                    "model_id": model_id,
                    "analysis_tier": point["analysis_tier"],
                    "metric": metric,
                    "point_estimate": point_estimate,
                    "bootstrap_standard_error": standard_error,
                    "confidence_level": protocol.confidence_level,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "finite_bootstrap_replicates": int(finite.numel()),
                    "bootstrap_interval_estimable": finite.numel() > 0,
                }
            )
            bootstrap_index.append(
                {"bootstrap_index": index, "model_id": model_id, "metric": metric, "tensor_row": index}
            )
            bootstrap_tensors.append(values.to(torch.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_metrics": output_dir / "model_metrics.csv",
        "split_metrics": output_dir / "split_metrics.csv",
        "reliability_bins": output_dir / "reliability_bins.csv",
        "high_confidence_errors": output_dir / "high_confidence_errors.csv",
        "risk_coverage": output_dir / "risk_coverage.csv",
        "bootstrap_ci": output_dir / "bootstrap_ci.csv",
        "bootstrap_index": output_dir / "bootstrap_replicate_index.csv",
        "bootstrap_replicates": output_dir / "bootstrap_replicates.pt",
    }
    for name, rows in (
        ("model_metrics", model_metric_rows),
        ("split_metrics", split_metric_rows),
        ("reliability_bins", reliability_rows),
        ("high_confidence_errors", high_confidence_rows),
        ("risk_coverage", risk_coverage_rows),
        ("bootstrap_ci", bootstrap_rows),
        ("bootstrap_index", bootstrap_index),
    ):
        _write_csv(outputs[name], rows)
    torch.save(
        {
            "schema_version": 1,
            "shape": [len(bootstrap_tensors), protocol.bootstrap_replicates],
            "statistics": torch.stack(bootstrap_tensors),
        },
        outputs["bootstrap_replicates"],
    )
    if protocol_path is not None:
        source_paths.append(protocol_path)
    manifest = {
        "schema_version": 1,
        "experiment_id": "C",
        "analysis_type": "uncalibrated_confidence_calibration_and_selective_prediction",
        "probability_postprocessing": "none",
        "design": {
            "model_ids": list(protocol.model_ids),
            "primary_model_ids": list(protocol.primary_model_ids),
            "supplementary_model_ids": list(protocol.supplementary_model_ids),
            "excluded_model_ids": ["M4"],
            "split_ids": list(split_ids),
            "shot": protocol.shot,
            "ece_bin_count": protocol.ece_bin_count,
            "high_confidence_thresholds": list(protocol.high_confidence_thresholds),
        },
        "bootstrap": {
            "hierarchy": ["class_split", "disease_class", "patient"],
            "replicates": protocol.bootstrap_replicates,
            "confidence_level": protocol.confidence_level,
            "seed": protocol.bootstrap_seed,
        },
        "source_artifacts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sorted(source_paths, key=str)
        ],
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "interpretation_limit": "Model probabilities are uncalibrated scores and must not be equated with clinical diagnostic confidence.",
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs/results"))
    parser.add_argument("--config", type=Path, default=Path("configs/calibration_analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_c_calibration"))
    args = parser.parse_args(argv)
    protocol = load_calibration_protocol(args.config)
    result = analyse_calibration(results_root=args.results_root, output_dir=args.output_dir, protocol=protocol, protocol_path=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
