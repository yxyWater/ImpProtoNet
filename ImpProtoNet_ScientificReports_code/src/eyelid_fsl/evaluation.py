"""Shared final-test evaluation and raw-result export."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .models.logistic import LogisticRegressionEpisodeModel
from .models.matchingnet import MatchingNet
from .models.protonet import ProtoNet
from .models.relationnet import RelationNet
from .training import collect_environment, episode_macro_f1


@dataclass(frozen=True)
class EvaluationIdentity:
    """Model identity fields repeated in every raw prediction record."""

    model_id: str
    model_name: str
    model_version: str
    training_seed: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stable_id(prefix: str, payload: object) -> str:
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:20]}"


def _git_provenance(project_root: Path | None) -> dict[str, object]:
    if project_root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _config_provenance(paths: Sequence[Path]) -> dict[str, object]:
    artifacts = [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in paths
    ]
    bundle_sha256 = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"artifacts": artifacts, "bundle_sha256": bundle_sha256}


class _FeatureShardWriter:
    """Stream optional high-dimensional mechanism tensors into bounded shards."""

    def __init__(self, output_dir: Path, shard_size: int) -> None:
        if shard_size < 1:
            raise ValueError("Feature shard size must be positive")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict[str, object]] = []
        self.shards: list[dict[str, object]] = []

    def append(self, payload: dict[str, object]) -> None:
        self.buffer.append(payload)
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_index = len(self.shards) + 1
        path = self.output_dir / f"feature_tensors_{shard_index:04d}.pt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.buffer, temporary)
        os.replace(temporary, path)
        self.shards.append(
            {
                "path": path.name,
                "episode_count": len(self.buffer),
                "sha256": _sha256(path),
            }
        )
        self.buffer = []

    def close(self) -> dict[str, object]:
        self._flush()
        manifest = {
            "schema_version": 1,
            "tensor_dtype": "float32",
            "shard_size_episodes": self.shard_size,
            "shards": self.shards,
        }
        _write_json(self.output_dir / "index.json", manifest)
        return manifest


class _EpisodeFeatureCache:
    """Keep deterministic evaluation encoder outputs in CPU memory for one run."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[tuple[str, str, str], torch.Tensor] = {}
        self.requested_feature_lookups = 0
        self.cache_misses = 0
        self.encoder_batches = 0

    @staticmethod
    def _key(
        namespace: str,
        metadata: dict[str, object],
    ) -> tuple[str, str, str]:
        return (
            namespace,
            str(metadata["image_id"]),
            str(metadata["source_relpath"]),
        )

    def encode(
        self,
        images: torch.Tensor,
        metadata: Sequence[dict[str, object]],
        encoder: torch.nn.Module,
        namespace: str,
    ) -> torch.Tensor:
        if images.shape[0] != len(metadata):
            raise ValueError("Image tensors and cache metadata have different lengths")
        keys = [self._key(namespace, item) for item in metadata]
        missing_indices: list[int] = []
        scheduled: set[tuple[str, str, str]] = set()
        for index, key in enumerate(keys):
            self.requested_feature_lookups += 1
            if key not in self.values and key not in scheduled:
                missing_indices.append(index)
                scheduled.add(key)
                self.cache_misses += 1
        if missing_indices:
            batch = images[missing_indices].to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            encoded = encoder(batch)
            if encoded.shape[0] != len(missing_indices):
                raise RuntimeError("Encoder changed the cached image batch size")
            for index, feature in zip(missing_indices, encoded.detach().cpu()):
                self.values[keys[index]] = feature.contiguous()
            self.encoder_batches += 1
        return torch.stack([self.values[key] for key in keys]).to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )

    def forward(
        self,
        model: torch.nn.Module,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        support_metadata: Sequence[dict[str, object]],
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        query_metadata: Sequence[dict[str, object]],
    ) -> object:
        if isinstance(model, ProtoNet):
            support_features = self.encode(
                support_images,
                support_metadata,
                model.encoder,
                "protonet_support_encoder",
            )
            query_features = self.encode(
                query_images,
                query_metadata,
                model.encoder,
                "protonet_query_encoder",
            )
            return model.forward_from_features(
                support_features, support_labels, query_features, query_labels
            )
        if isinstance(model, LogisticRegressionEpisodeModel):
            support_features = self.encode(
                support_images,
                support_metadata,
                model.encoder,
                "logistic_support_encoder",
            )
            query_features = self.encode(
                query_images,
                query_metadata,
                model.encoder,
                "logistic_query_encoder",
            )
            return model.forward_from_features(
                support_features, support_labels, query_features, query_labels
            )
        if isinstance(model, MatchingNet):
            base_support = self.encode(
                support_images,
                support_metadata,
                model.support_encoder,
                "matching_support_encoder",
            )
            query_encoder = (
                model.support_encoder
                if model.encoders_are_shared
                else model.query_encoder
            )
            base_query = self.encode(
                query_images,
                query_metadata,
                query_encoder,
                "matching_query_encoder",
            )
            return model.forward_from_features(
                base_support, support_labels, base_query, query_labels
            )
        if isinstance(model, RelationNet):
            support_maps = self.encode(
                support_images,
                support_metadata,
                model.spatial_encoder,
                "relation_spatial_encoder",
            )
            query_maps = self.encode(
                query_images,
                query_metadata,
                model.spatial_encoder,
                "relation_spatial_encoder",
            )
            return model.forward_from_features(
                support_maps, support_labels, query_maps, query_labels
            )
        raise TypeError(
            "Image-feature caching supports the approved M0-M5 episode models only"
        )

    def summary(self) -> dict[str, object]:
        hits = self.requested_feature_lookups - self.cache_misses
        return {
            "enabled": True,
            "scope": "current_evaluation_run_and_model_checkpoint_only",
            "storage_device": "cpu",
            "requested_feature_lookups": self.requested_feature_lookups,
            "cache_hits": hits,
            "cache_misses": self.cache_misses,
            "unique_cached_encoder_outputs": len(self.values),
            "encoder_batches": self.encoder_batches,
            "hit_rate": (
                hits / self.requested_feature_lookups
                if self.requested_feature_lookups
                else 0.0
            ),
        }


def load_best_protonet_checkpoint(
    model: ProtoNet,
    checkpoint_path: Path | None,
    *,
    expected_split_id: str,
    expected_shot: int,
    expected_training_seed: int | None = None,
    expected_model_id: str | None = None,
    expected_learning_rate: float | None = None,
) -> dict[str, object]:
    """Load a validation-selected checkpoint and reject experiment mismatches."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("split_id") != expected_split_id:
        raise ValueError("Checkpoint belongs to a different class split")
    if int(payload.get("shot", -1)) != expected_shot:
        raise ValueError("Checkpoint belongs to a different shot configuration")
    if payload.get("best_validation") is None:
        raise ValueError("Checkpoint has no validation-selected model state")
    if (
        expected_training_seed is not None
        and int(payload.get("training_seed", -1)) != expected_training_seed
    ):
        raise ValueError("Checkpoint uses a different training seed")
    if (
        expected_model_id is not None
        and payload.get("model_id") != expected_model_id
    ):
        raise ValueError("Checkpoint belongs to a different model")
    if (
        expected_learning_rate is not None
        and not math.isclose(
            float(payload.get("learning_rate", float("nan"))),
            expected_learning_rate,
        )
    ):
        raise ValueError("Checkpoint uses a different learning rate")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _sample_standard_deviation(values: Sequence[float]) -> float:
    return 0.0 if len(values) < 2 else statistics.stdev(values)


def evaluate_test_run(
    model: torch.nn.Module,
    episodes: Iterable[dict[str, object]],
    *,
    device: torch.device,
    output_dir: Path,
    checkpoint_path: Path | None,
    episode_manifest_path: Path,
    mixed_precision: bool,
    identity: EvaluationIdentity | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    dataset_manifest_path: Path | None = None,
    config_paths: Sequence[Path] = (),
    project_root: Path | None = None,
    save_feature_tensors: bool = False,
    feature_shard_size: int = 100,
    expected_partition: str = "test",
    result_prefix: str = "test",
    cache_image_features: bool = False,
    expected_episode_count: int | None = None,
) -> dict[str, object]:
    """Evaluate one fixed episode manifest and write unaggregated results."""

    if expected_partition not in {"validation", "test"}:
        raise ValueError("Evaluation partition must be validation or test")
    if result_prefix not in {"validation", "test"}:
        raise ValueError("Result prefix must be validation or test")
    if result_prefix != expected_partition:
        raise ValueError("Result prefix must match the evaluated partition")

    model.to(device)
    model.eval()
    query_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    class_names: dict[str, str] = {}
    class_categories: dict[str, str] = {}
    run_split_id: str | None = None
    run_shot: int | None = None
    identity = identity or EvaluationIdentity("M0", "ProtoNet", "1", None)
    checkpoint_metadata = checkpoint_metadata or {}
    checkpoint_sha256 = (
        _sha256(checkpoint_path) if checkpoint_path is not None else None
    )
    model_state_sha256 = _model_state_sha256(model)
    dataset_manifest_sha256 = (
        _sha256(dataset_manifest_path) if dataset_manifest_path is not None else None
    )
    config_provenance = _config_provenance(config_paths)
    git_provenance = _git_provenance(project_root)
    feature_writer = (
        _FeatureShardWriter(
            output_dir / f"{result_prefix}_feature_tensors", feature_shard_size
        )
        if save_feature_tensors
        else None
    )
    support_set_rows: list[dict[str, object]] = []
    feature_cache = _EpisodeFeatureCache(device) if cache_image_features else None

    with torch.no_grad():
        for raw_episode in episodes:
            if raw_episode["partition"] != expected_partition:
                raise ValueError(
                    f"Evaluation accepts {expected_partition} episodes only"
                )
            episode_split_id = str(raw_episode["split_id"])
            episode_shot = int(raw_episode["shot"])
            if run_split_id is None:
                run_split_id = episode_split_id
                run_shot = episode_shot
            elif episode_split_id != run_split_id or episode_shot != run_shot:
                raise ValueError(
                    "Final evaluation cannot mix class splits or shot configurations"
                )
            support_metadata = tuple(raw_episode["support_metadata"])
            query_metadata = tuple(raw_episode["query_metadata"])
            support_images = raw_episode["support_images"]
            query_images = raw_episode["query_images"]
            if feature_cache is None:
                support_images = support_images.to(
                    device, non_blocking=device.type == "cuda"
                )
                query_images = query_images.to(
                    device, non_blocking=device.type == "cuda"
                )
            support_labels = raw_episode["support_labels"].to(
                device, non_blocking=device.type == "cuda"
            )
            query_labels = raw_episode["query_labels"].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
                if feature_cache is None:
                    output = model(
                        support_images,
                        support_labels,
                        query_images,
                        query_labels,
                    )
                else:
                    output = feature_cache.forward(
                        model,
                        support_images,
                        support_labels,
                        support_metadata,
                        query_images,
                        query_labels,
                        query_metadata,
                    )

            predictions = output.predictions.detach().cpu()
            targets = query_labels.detach().cpu()
            probabilities = output.probabilities.detach().float().cpu()
            distances = output.squared_distances.detach().float().cpu()
            logits = output.logits.detach().float().cpu()
            losses = output.per_query_loss.detach().float().cpu()
            support_features = output.support_features.detach().float().cpu()
            query_features = output.query_features.detach().float().cpu()
            prototypes = output.prototypes.detach().float().cpu()
            matching_attention = (
                output.support_attention.detach().float().cpu()
                if hasattr(output, "support_attention")
                else None
            )
            matching_cosine_similarities = (
                output.cosine_similarities.detach().float().cpu()
                if hasattr(output, "cosine_similarities")
                else None
            )
            relation_scores = (
                output.relation_scores.detach().float().cpu()
                if hasattr(output, "relation_scores")
                else None
            )
            support_labels_cpu = support_labels.detach().cpu()
            class_ids = tuple(str(item) for item in raw_episode["class_ids"])
            names = tuple(str(item) for item in raw_episode["class_names"])
            categories = tuple(str(item) for item in raw_episode["class_categories"])
            for class_id, name, category in zip(class_ids, names, categories):
                class_names[class_id] = name
                class_categories[class_id] = category

            experiment_context = {
                key: raw_episode[key]
                for key in (
                    "experiment_id",
                    "disease_combination_index",
                    "query_panel_index",
                    "support_resample_index",
                    "maximum_support_resamples",
                    "evaluated_support_resample_count",
                    "fixed_image_per_patient",
                    "replacement_pair_index",
                    "replacement_pair_member",
                    "replaced_class_id",
                    "replaced_support_rank",
                    "replaced_patient_id",
                    "replacement_patient_id",
                )
                if key in raw_episode
            }
            disease_combination_id = _stable_id(
                "disease_combo",
                {"split_id": episode_split_id, "class_ids": class_ids},
            )
            query_panel_id = _stable_id(
                "query_panel",
                {
                    "split_id": episode_split_id,
                    "class_ids": class_ids,
                    "members": [
                        {
                            "class_id": item["class_id"],
                            "patient_id": item["patient_id"],
                            "image_id": item["image_id"],
                        }
                        for item in query_metadata
                    ],
                },
            )

            def support_identifier(
                selected_shot: int,
                members: Sequence[dict[str, object]],
            ) -> str:
                canonical_members = sorted(
                    members,
                    key=lambda item: (
                        str(item["class_id"]),
                        str(item["patient_id"]),
                        str(item["image_id"]),
                        int(item["support_rank"]),
                    ),
                )
                return _stable_id(
                    "support_set",
                    {
                        "split_id": episode_split_id,
                        "class_ids": class_ids,
                        "shot": selected_shot,
                        "members": [
                            {
                                "class_id": item["class_id"],
                                "patient_id": item["patient_id"],
                                "image_id": item["image_id"],
                                "support_rank": item["support_rank"],
                            }
                            for item in canonical_members
                        ],
                    },
                )

            support_set_id = support_identifier(episode_shot, support_metadata)
            nested_three_members = tuple(
                item for item in support_metadata if int(item["support_rank"]) <= 3
            )
            nested_three_shot_support_set_id = support_identifier(
                3,
                nested_three_members,
            )
            prototype_dispersion: dict[str, float] = {}
            prototype_distances: dict[str, dict[str, float]] = {}
            for class_slot, class_id in enumerate(class_ids):
                class_support = support_features[support_labels_cpu == class_slot]
                prototype_dispersion[class_id] = float(
                    (class_support - prototypes[class_slot])
                    .square()
                    .sum(dim=1)
                    .mean()
                )
                prototype_distances[class_id] = {
                    other_class_id: float(
                        (prototypes[class_slot] - prototypes[other_slot]).square().sum()
                    )
                    for other_slot, other_class_id in enumerate(class_ids)
                }
            fixed_query_panel = bool(raw_episode.get("fixed_query_panel", False))
            support_set_rows.append(
                {
                    "episode_id": raw_episode["episode_id"],
                    "split_id": episode_split_id,
                    "partition": raw_episode["partition"],
                    "disease_combination_id": disease_combination_id,
                    "query_panel_id": query_panel_id,
                    "fixed_query_panel": fixed_query_panel,
                    "support_set_id": support_set_id,
                    "nested_3shot_support_set_id": nested_three_shot_support_set_id,
                    "shot": episode_shot,
                    "class_ids": class_ids,
                    "support_members": [dict(item) for item in support_metadata],
                    "prototype_dispersion_by_class": prototype_dispersion,
                    "prototype_squared_distances_by_class": prototype_distances,
                    **experiment_context,
                }
            )
            if feature_writer is not None:
                feature_payload = {
                    "episode_id": raw_episode["episode_id"],
                    "support_set_id": support_set_id,
                    "query_panel_id": query_panel_id,
                    "class_ids": class_ids,
                    "support_features": support_features,
                    "support_labels": support_labels_cpu,
                    "query_features": query_features,
                    "query_labels": targets,
                    "prototypes": prototypes,
                }
                if hasattr(output, "base_support_features"):
                    feature_payload.update(
                        {
                            "base_support_features": (
                                output.base_support_features.detach().float().cpu()
                            ),
                            "base_query_features": (
                                output.base_query_features.detach().float().cpu()
                            ),
                            "matching_support_attention": matching_attention,
                            "matching_cosine_similarities": (
                                matching_cosine_similarities
                            ),
                        }
                    )
                if hasattr(output, "support_feature_maps"):
                    feature_payload.update(
                        {
                            "relation_support_feature_maps": (
                                output.support_feature_maps.detach().float().cpu()
                            ),
                            "relation_query_feature_maps": (
                                output.query_feature_maps.detach().float().cpu()
                            ),
                            "relation_class_support_feature_maps": (
                                output.class_support_feature_maps.detach().float().cpu()
                            ),
                            "relation_scores": relation_scores,
                        }
                    )
                feature_writer.append(feature_payload)

            accuracy = float((predictions == targets).float().mean())
            macro_f1 = episode_macro_f1(predictions, targets, model.way)
            episode_rows.append(
                {
                    "episode_id": raw_episode["episode_id"],
                    "split_id": raw_episode["split_id"],
                    "partition": raw_episode["partition"],
                    "shot": raw_episode["shot"],
                    "disease_combination_id": disease_combination_id,
                    "query_panel_id": query_panel_id,
                    "support_set_id": support_set_id,
                    "class_ids": "|".join(class_ids),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "mean_loss": float(losses.mean()),
                    **experiment_context,
                }
            )
            for query_index, metadata in enumerate(query_metadata):
                target_label = int(targets[query_index])
                predicted_label = int(predictions[query_index])
                target_class_id = class_ids[target_label]
                predicted_class_id = class_ids[predicted_label]
                if str(metadata["class_id"]) != target_class_id:
                    raise ValueError(
                        f"Query metadata/label mismatch in {raw_episode['episode_id']}"
                    )
                sorted_probabilities = probabilities[query_index].sort(descending=True).values
                query_row = {
                        "schema_version": 2,
                        "model_id": identity.model_id,
                        "model_name": identity.model_name,
                        "model_version": identity.model_version,
                        "training_seed": identity.training_seed,
                        "classification_rule": checkpoint_metadata.get(
                            "classification_rule"
                        ),
                        "regularization_c": checkpoint_metadata.get(
                            "regularization_c"
                        ),
                        "checkpoint_sha256": checkpoint_sha256,
                        "model_state_sha256": model_state_sha256,
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "experiment_config_bundle_sha256": config_provenance["bundle_sha256"],
                        "code_commit": git_provenance["commit"],
                        "episode_id": raw_episode["episode_id"],
                        "split_id": raw_episode["split_id"],
                        "partition": raw_episode["partition"],
                        "shot": raw_episode["shot"],
                        "disease_combination_id": disease_combination_id,
                        "query_panel_id": query_panel_id,
                        "fixed_query_panel": fixed_query_panel,
                        "support_set_id": support_set_id,
                        "nested_3shot_support_set_id": nested_three_shot_support_set_id,
                        "query_index": query_index,
                        "query_rank": metadata.get("query_rank"),
                        "patient_id": metadata["patient_id"],
                        "image_id": metadata["image_id"],
                        "source_relpath": metadata["source_relpath"],
                        "target_class_id": target_class_id,
                        "target_class_name": class_names[target_class_id],
                        "target_class_slot": target_label + 1,
                        "predicted_class_id": predicted_class_id,
                        "predicted_class_name": class_names[predicted_class_id],
                        "predicted_class_slot": predicted_label + 1,
                        "correct": target_class_id == predicted_class_id,
                        "confidence": float(probabilities[query_index].max()),
                        "true_class_probability": float(
                            probabilities[query_index, target_label]
                        ),
                        "top1_top2_probability_margin": float(
                            sorted_probabilities[0] - sorted_probabilities[1]
                        ),
                        "query_loss": float(losses[query_index]),
                        "logits_by_class": {
                            class_id: float(logits[query_index, slot])
                            for slot, class_id in enumerate(class_ids)
                        },
                        "probabilities_by_class": {
                            class_id: float(probabilities[query_index, slot])
                            for slot, class_id in enumerate(class_ids)
                        },
                        "squared_distances_by_class": {
                            class_id: float(distances[query_index, slot])
                            for slot, class_id in enumerate(class_ids)
                        },
                        **experiment_context,
                    }
                if matching_attention is not None:
                    query_row.update(
                        {
                            "matching_support_attention": matching_attention[
                                query_index
                            ].tolist(),
                            "matching_cosine_similarities": matching_cosine_similarities[
                                query_index
                            ].tolist(),
                            "matching_support_array_alignment": (
                                f"{result_prefix}_support_sets.support_members"
                            ),
                        }
                    )
                if relation_scores is not None:
                    query_row["relation_scores_by_class"] = {
                        class_id: float(relation_scores[query_index, slot])
                        for slot, class_id in enumerate(class_ids)
                    }
                query_rows.append(query_row)

    if not episode_rows:
        raise ValueError("Evaluation loader produced no episodes")
    if expected_episode_count is not None and len(episode_rows) != expected_episode_count:
        raise ValueError(
            f"Evaluation produced {len(episode_rows)} episodes; "
            f"{expected_episode_count} required"
        )
    all_class_ids = sorted(class_names)
    confusion = Counter(
        (row["target_class_id"], row["predicted_class_id"]) for row in query_rows
    )
    per_class_rows: list[dict[str, object]] = []
    for class_id in all_class_ids:
        true_positive = confusion[(class_id, class_id)]
        false_positive = sum(
            confusion[(other_id, class_id)]
            for other_id in all_class_ids
            if other_id != class_id
        )
        false_negative = sum(
            confusion[(class_id, other_id)]
            for other_id in all_class_ids
            if other_id != class_id
        )
        support = true_positive + false_negative
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, support)
        f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "category": class_categories[class_id],
                "query_count": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    confusion_rows = [
        {
            "actual_class_id": actual_id,
            "actual_class_name": class_names[actual_id],
            "predicted_class_id": predicted_id,
            "predicted_class_name": class_names[predicted_id],
            "count": confusion[(actual_id, predicted_id)],
        }
        for actual_id in all_class_ids
        for predicted_id in all_class_ids
    ]

    episode_accuracies = [float(row["accuracy"]) for row in episode_rows]
    episode_macro_f1s = [float(row["macro_f1"]) for row in episode_rows]
    episode_losses = [float(row["mean_loss"]) for row in episode_rows]
    pooled_accuracy = sum(bool(row["correct"]) for row in query_rows) / len(query_rows)
    pooled_macro_f1 = sum(float(row["f1"]) for row in per_class_rows) / len(per_class_rows)
    balanced_accuracy = sum(float(row["recall"]) for row in per_class_rows) / len(per_class_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(
        output_dir / f"{result_prefix}_query_predictions.jsonl.gz", query_rows
    )
    _write_jsonl_gz(
        output_dir / f"{result_prefix}_support_sets.jsonl.gz", support_set_rows
    )
    _write_csv(output_dir / f"{result_prefix}_episode_metrics.csv", episode_rows)
    _write_csv(output_dir / f"{result_prefix}_per_class_metrics.csv", per_class_rows)
    _write_csv(
        output_dir / f"{result_prefix}_confusion_matrix.csv", confusion_rows
    )
    feature_manifest = feature_writer.close() if feature_writer is not None else None
    feature_cache_summary = (
        feature_cache.summary() if feature_cache is not None else {"enabled": False}
    )
    run_manifest = {
        "schema_version": 2,
        "model": {
            "model_id": identity.model_id,
            "model_name": identity.model_name,
            "model_version": identity.model_version,
            "training_seed": identity.training_seed,
        },
        "dataset_manifest": (
            {
                "path": str(dataset_manifest_path.resolve()),
                "sha256": dataset_manifest_sha256,
            }
            if dataset_manifest_path is not None
            else None
        ),
        "episode_manifest": {
            "path": str(episode_manifest_path.resolve()),
            "sha256": _sha256(episode_manifest_path),
        },
        "model_artifact": {
            "type": "checkpoint" if checkpoint_path is not None else "pretrained_weights",
            "path": (
                str(checkpoint_path.resolve()) if checkpoint_path is not None else None
            ),
            "file_sha256": checkpoint_sha256,
            "model_state_sha256": model_state_sha256,
            "metadata": {
                "learning_rate": checkpoint_metadata.get("learning_rate"),
                "training_seed": checkpoint_metadata.get("training_seed"),
                "completed_episode": checkpoint_metadata.get("completed_episode"),
                "completed_epoch": checkpoint_metadata.get("completed_epoch"),
                "best_validation": checkpoint_metadata.get("best_validation"),
                "classification_rule": checkpoint_metadata.get(
                    "classification_rule"
                ),
                "regularization_c": checkpoint_metadata.get("regularization_c"),
                "matching_processing_steps": checkpoint_metadata.get(
                    "matching_processing_steps"
                ),
                "matching_support_order_seed": checkpoint_metadata.get(
                    "matching_support_order_seed"
                ),
                "relation_support_aggregation": checkpoint_metadata.get(
                    "relation_support_aggregation"
                ),
                "relation_training_loss": checkpoint_metadata.get(
                    "relation_training_loss"
                ),
                "relation_probability_rule": checkpoint_metadata.get(
                    "relation_probability_rule"
                ),
                "weights_enum": checkpoint_metadata.get("weights_enum"),
                "weights_url": checkpoint_metadata.get("weights_url"),
                "evaluated_support_resample_count": checkpoint_metadata.get(
                    "evaluated_support_resample_count"
                ),
                "support_resample_selection_sha256": checkpoint_metadata.get(
                    "support_resample_selection_sha256"
                ),
                "single_replacement_pairs_per_combination": checkpoint_metadata.get(
                    "single_replacement_pairs_per_combination"
                ),
            },
        },
        "experiment_configs": config_provenance,
        "code": git_provenance,
        "feature_tensors": feature_manifest,
        "image_feature_cache": feature_cache_summary,
    }
    _write_json(output_dir / f"{result_prefix}_run_manifest.json", run_manifest)
    summary = {
        "schema_version": 2,
        "model_id": identity.model_id,
        "model_name": identity.model_name,
        "model_version": identity.model_version,
        "training_seed": identity.training_seed,
        "split_id": run_split_id,
        "shot": run_shot,
        "checkpoint_path": (
            str(checkpoint_path.resolve()) if checkpoint_path is not None else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "episode_manifest_path": str(episode_manifest_path.resolve()),
        "episode_manifest_sha256": _sha256(episode_manifest_path),
        "environment": collect_environment(device),
        "mixed_precision_enabled": mixed_precision,
        "image_feature_cache": feature_cache_summary,
        "episode_level": {
            "episode_count": len(episode_rows),
            "mean_accuracy": statistics.mean(episode_accuracies),
            "accuracy_sd": _sample_standard_deviation(episode_accuracies),
            "minimum_accuracy": min(episode_accuracies),
            "maximum_accuracy": max(episode_accuracies),
            "mean_macro_f1": statistics.mean(episode_macro_f1s),
            "macro_f1_sd": _sample_standard_deviation(episode_macro_f1s),
            "mean_loss": statistics.mean(episode_losses),
        },
        "pooled_query_level": {
            "query_prediction_count": len(query_rows),
            "accuracy": pooled_accuracy,
            "macro_f1_by_disease": pooled_macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "mean_query_loss": statistics.mean(
                float(row["query_loss"]) for row in query_rows
            ),
        },
        "statistical_note": (
            "Repeated episode predictions are not independent observations; "
            "confidence intervals and hypothesis tests are deferred to the grouped statistical module."
        ),
    }
    _write_json(output_dir / f"{result_prefix}_summary.json", summary)
    return summary
