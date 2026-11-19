"""Patient-balanced supervised samples for the M2 encoder training stage."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .episodes import ClassPool, SplitImage
from .imaging import ImagePreprocessor, deterministic_horizontal_flip


@dataclass(frozen=True)
class SupervisedSample:
    """One patient-level draw used for ordinary supervised fine-tuning."""

    sample_index: int
    epoch: int
    class_id: str
    class_label: int
    patient_id: str
    image_id: str
    source_relpath: str


def _derived_seed(master_seed: int, *parts: object) -> int:
    material = "|".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _balanced_class_sequence(
    class_ids: Sequence[str],
    sample_count: int,
    rng: random.Random,
) -> list[str]:
    if sample_count < len(class_ids):
        raise ValueError("sample_count must expose every training class in each epoch")
    full_cycles, remainder = divmod(sample_count, len(class_ids))
    result: list[str] = []
    for _ in range(full_cycles):
        cycle = list(class_ids)
        rng.shuffle(cycle)
        result.extend(cycle)
    partial = list(class_ids)
    rng.shuffle(partial)
    result.extend(partial[:remainder])
    return result


def build_supervised_epoch_samples(
    class_pools: Mapping[str, ClassPool],
    *,
    epoch: int,
    master_seed: int,
    sample_count: int | None = None,
) -> tuple[SupervisedSample, ...]:
    """Draw disease, then patient, then one image with reproducible balancing.

    Disease counts differ by at most one. Within each disease, every patient is
    visited once in a shuffled cycle before any patient begins the next cycle.
    A patient with several photographs therefore receives no more sampling
    weight than a patient with one photograph.
    """

    if epoch < 1:
        raise ValueError("epoch must be positive")
    if not class_pools:
        raise ValueError("At least one training class is required")
    class_ids = tuple(sorted(class_pools))
    if any(not class_pools[class_id].images_by_patient for class_id in class_ids):
        raise ValueError("Every training class must contain at least one patient")
    unique_patient_count = sum(
        len(class_pools[class_id].images_by_patient) for class_id in class_ids
    )
    if sample_count is None:
        sample_count = unique_patient_count
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    rng = random.Random(_derived_seed(master_seed, "supervised", epoch))
    class_sequence = _balanced_class_sequence(class_ids, sample_count, rng)
    labels = {class_id: index for index, class_id in enumerate(class_ids)}
    patient_queues: dict[str, deque[str]] = {
        class_id: deque() for class_id in class_ids
    }

    def next_patient(class_id: str) -> str:
        queue = patient_queues[class_id]
        if not queue:
            patient_ids = sorted(class_pools[class_id].images_by_patient)
            rng.shuffle(patient_ids)
            queue.extend(patient_ids)
        return queue.popleft()

    samples: list[SupervisedSample] = []
    for sample_index, class_id in enumerate(class_sequence):
        patient_id = next_patient(class_id)
        patient_images: tuple[SplitImage, ...] = class_pools[
            class_id
        ].images_by_patient[patient_id]
        image = rng.choice(patient_images)
        samples.append(
            SupervisedSample(
                sample_index=sample_index,
                epoch=epoch,
                class_id=class_id,
                class_label=labels[class_id],
                patient_id=patient_id,
                image_id=image.image_id,
                source_relpath=image.source_relpath,
            )
        )

    class_counts = Counter(sample.class_id for sample in samples)
    if max(class_counts.values()) - min(class_counts.values()) > 1:
        raise RuntimeError("Internal error: disease schedule is not balanced")
    return tuple(samples)


class SupervisedPatientDataset(Dataset):
    """Materialise a fixed, auditable supervised epoch schedule."""

    def __init__(
        self,
        samples: Sequence[SupervisedSample],
        source_root: Path,
        *,
        augmentation_seed: int,
        horizontal_flip_probability: float = 0.5,
        image_size: int = 224,
    ) -> None:
        self.samples = tuple(samples)
        self.source_root = Path(source_root)
        self.augmentation_seed = int(augmentation_seed)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        if not self.samples:
            raise ValueError("SupervisedPatientDataset cannot be empty")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("Horizontal flip probability must be between zero and one")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        should_flip = deterministic_horizontal_flip(
            self.horizontal_flip_probability,
            self.augmentation_seed,
            "supervised",
            sample.epoch,
            sample.sample_index,
            sample.image_id,
        )
        image_path = self.source_root / Path(sample.source_relpath)
        if not image_path.is_file():
            raise FileNotFoundError(f"Supervised image does not exist: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.preprocessor(image, horizontal_flip=should_flip)
        return {
            "image": tensor,
            "label": torch.tensor(sample.class_label, dtype=torch.long),
            "sample_index": sample.sample_index,
            "epoch": sample.epoch,
            "class_id": sample.class_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "source_relpath": sample.source_relpath,
            "horizontal_flipped": should_flip,
        }


def build_supervised_dataloader(
    dataset: SupervisedPatientDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
    seed: int,
) -> DataLoader:
    """Create a deterministic loader; sampling order already lives in the dataset."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )
