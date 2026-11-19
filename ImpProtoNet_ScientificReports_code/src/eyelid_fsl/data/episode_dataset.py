"""PyTorch loading of fixed patient-level episode manifests."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .episodes import read_episode_manifest
from .imaging import ImagePreprocessor, deterministic_horizontal_flip


@lru_cache(maxsize=32)
def _cached_episode_manifest(manifest_path: str) -> tuple[dict[str, object], ...]:
    """Parse each compressed manifest once per worker process."""
    return tuple(read_episode_manifest(Path(manifest_path)))


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve CUDA when available and otherwise fall back to CPU."""

    normalised = requested.casefold()
    if normalised == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalised)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment")
    return device


class EpisodeTensorDataset(IterableDataset):
    """Stream one complete few-shot episode at a time from a compressed manifest."""

    def __init__(
        self,
        manifest_path: Path,
        source_root: Path,
        shot: int,
        *,
        training: bool,
        augmentation_seed: int = 20261004,
        horizontal_flip_probability: float = 0.5,
        image_size: int = 224,
        support_order_seed: int | None = None,
        maximum_support_resample_index: int | None = None,
        image_cache_size: int = 256,
        color_jitter_strength: float = 0.0,
        rotation_degrees: float = 0.0,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.source_root = Path(source_root)
        self.shot = int(shot)
        self.training = bool(training)
        self.augmentation_seed = int(augmentation_seed)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.support_order_seed = (
            None if support_order_seed is None else int(support_order_seed)
        )
        self.maximum_support_resample_index = (
            None
            if maximum_support_resample_index is None
            else int(maximum_support_resample_index)
        )
        self.color_jitter_strength = float(color_jitter_strength)
        self.rotation_degrees = float(rotation_degrees)
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        if image_cache_size < 0:
            raise ValueError("image_cache_size cannot be negative")
        self.image_cache_size = int(image_cache_size)
        if self.shot < 1:
            raise ValueError("shot must be positive")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("Horizontal flip probability must be between zero and one")
        if (
            self.maximum_support_resample_index is not None
            and self.maximum_support_resample_index < 1
        ):
            raise ValueError("Maximum support resample index must be positive")

    @lru_cache(maxsize=256)
    def _cached_image(self, path_string, should_flip, cj_key="", rotation_deg=0.0):
        path = Path(path_string)
        with Image.open(path) as image:
            cj_params = None
            if cj_key:
                parts = cj_key.split(",")
                if len(parts) == 4:
                    cj_params = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            return self.preprocessor(image, horizontal_flip=should_flip, cj_params=cj_params, rotation_deg=rotation_deg)

    def _load_image(
        self,
        item: dict[str, object],
        episode_id: str,
        role: str,
        class_slot: int,
        rank: int,
    ) -> torch.Tensor:
        should_flip = self.training and deterministic_horizontal_flip(
            self.horizontal_flip_probability,
            self.augmentation_seed,
            episode_id,
            role,
            class_slot,
            rank,
            item["image_id"],
        )
        cj_params = None
        rotation_deg = 0.0
        if self.training and self.color_jitter_strength > 0.0:
            from .imaging import deterministic_color_jitter
            cj_params = deterministic_color_jitter(
                self.color_jitter_strength,
                self.augmentation_seed,
                episode_id, role, class_slot, rank, item["image_id"],
            )
        if self.training and self.rotation_degrees > 0.0:
            from .imaging import deterministic_rotation
            rotation_deg = deterministic_rotation(
                self.rotation_degrees,
                self.augmentation_seed,
                episode_id, role, class_slot, rank, item["image_id"],
            )
        path = self.source_root / Path(str(item["source_relpath"]))
        if not path.is_file():
            raise FileNotFoundError(f"Episode image does not exist: {path}")
        cj_key = ""
        if cj_params is not None:
            cj_key = f"{cj_params[0]},{cj_params[1]},{cj_params[2]},{cj_params[3]}"
        if self.image_cache_size == 0:
            with Image.open(path) as image:
                return self.preprocessor(image, horizontal_flip=should_flip, cj_params=cj_params, rotation_deg=rotation_deg)
        return self._cached_image(str(path), bool(should_flip), cj_key, rotation_deg)

    def _materialise(self, episode: dict[str, object]) -> dict[str, object]:
        supported_shots = {int(item) for item in episode["support_shots"]}
        if self.shot not in supported_shots:
            raise ValueError(
                f"Shot {self.shot} is not present in episode {episode['episode_id']}"
            )

        support_images: list[torch.Tensor] = []
        support_labels: list[int] = []
        support_metadata: list[dict[str, object]] = []
        query_images: list[torch.Tensor] = []
        query_labels: list[int] = []
        query_metadata: list[dict[str, object]] = []
        episode_id = str(episode["episode_id"])

        for class_item in episode["classes"]:
            class_slot = int(class_item["class_slot"])
            label = class_slot - 1
            class_id = str(class_item["class_id"])
            for item in class_item["support"]:
                if self.shot not in {int(value) for value in item["used_in_shots"]}:
                    continue
                rank = int(item["support_rank"])
                support_images.append(
                    self._load_image(item, episode_id, "support", class_slot, rank)
                )
                support_labels.append(label)
                support_metadata.append(
                    {
                        "class_id": class_id,
                        "class_slot": class_slot,
                        "patient_id": item["patient_id"],
                        "image_id": item["image_id"],
                        "source_relpath": item["source_relpath"],
                        "support_rank": rank,
                        "used_in_shots": tuple(int(value) for value in item["used_in_shots"]),
                    }
                )
            for item in class_item["query"]:
                rank = int(item["query_rank"])
                query_images.append(
                    self._load_image(item, episode_id, "query", class_slot, rank)
                )
                query_labels.append(label)
                query_metadata.append(
                    {
                        "class_id": class_id,
                        "class_slot": class_slot,
                        "patient_id": item["patient_id"],
                        "image_id": item["image_id"],
                        "source_relpath": item["source_relpath"],
                        "query_rank": rank,
                    }
                )

        if not support_images or not query_images:
            raise ValueError(f"Episode {episode_id} contains no usable images")
        expected_support = int(episode["way"]) * self.shot
        expected_query = int(episode["way"]) * int(
            episode["query_patients_per_class"]
        )
        if len(support_images) != expected_support:
            raise ValueError(f"Episode {episode_id} has an invalid support image count")
        if len(query_images) != expected_query:
            raise ValueError(f"Episode {episode_id} has an invalid query image count")
        if self.support_order_seed is not None:
            def support_order_key(index: int) -> bytes:
                metadata = support_metadata[index]
                material = "|".join(
                    (
                        str(self.support_order_seed),
                        episode_id,
                        str(metadata["patient_id"]),
                        str(metadata["image_id"]),
                    )
                )
                return hashlib.sha256(material.encode("utf-8")).digest()

            order = sorted(range(len(support_images)), key=support_order_key)
            support_images = [support_images[index] for index in order]
            support_labels = [support_labels[index] for index in order]
            support_metadata = [support_metadata[index] for index in order]
        materialised = {
            "episode_id": episode_id,
            "split_id": episode["split_id"],
            "partition": episode["partition"],
            "shot": self.shot,
            "class_ids": tuple(episode["class_ids"]),
            "class_names": tuple(
                str(class_item["class_name"]) for class_item in episode["classes"]
            ),
            "class_categories": tuple(
                str(class_item["category"]) for class_item in episode["classes"]
            ),
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "support_metadata": tuple(support_metadata),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "query_metadata": tuple(query_metadata),
        }
        for key in (
            "experiment_id",
            "disease_combination_index",
            "query_panel_index",
            "support_resample_index",
            "maximum_support_resamples",
            "fixed_query_panel",
            "fixed_image_per_patient",
            "replacement_pair_index",
            "replacement_pair_member",
            "replaced_class_id",
            "replaced_support_rank",
            "replaced_patient_id",
            "replacement_patient_id",
        ):
            if key in episode:
                materialised[key] = episode[key]
        if self.maximum_support_resample_index is not None:
            materialised["evaluated_support_resample_count"] = (
                self.maximum_support_resample_index
            )
        return materialised

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker = get_worker_info()
        for index, episode in enumerate(_cached_episode_manifest(str(self.manifest_path))):
            if worker is not None and index % worker.num_workers != worker.id:
                continue
            if self.maximum_support_resample_index is not None:
                if "support_resample_index" not in episode:
                    raise ValueError(
                        "Support-resample filtering requires experiment B episodes"
                    )
                if (
                    int(episode["support_resample_index"])
                    > self.maximum_support_resample_index
                ):
                    continue
            yield self._materialise(episode)


def build_episode_dataloader(
    dataset: EpisodeTensorDataset,
    *,
    device: torch.device,
    num_workers: int = 4,
) -> DataLoader:
    """Build an episode-at-a-time loader suitable for ResNet18 training."""

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
