"""Create fixed patient-level episode manifests for few-shot experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class EpisodeProtocol:
    way: int
    support_shots: tuple[int, ...]
    query_patients_per_class: int
    episode_counts: dict[str, int]
    master_seed: int
    validation_interval_episodes: int
    minimum_train_episodes: int
    early_stopping_patience_checks: int
    maximum_train_episodes: int

    @property
    def maximum_shot(self) -> int:
        return max(self.support_shots)


@dataclass(frozen=True)
class SplitImage:
    split_id: str
    partition: str
    image_id: str
    class_id: str
    class_name: str
    category: str
    patient_id: str
    source_relpath: str


@dataclass
class ClassPool:
    class_id: str
    class_name: str
    category: str
    images_by_patient: dict[str, tuple[SplitImage, ...]]


def load_episode_protocol(path: Path) -> EpisodeProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sampling = payload["sampling"]
    if sampling != {
        "class_sampling": "balanced_combinations",
        "patient_sampling": "uniform_without_replacement",
        "image_sampling": "one_uniform_image_per_selected_patient",
        "nested_support_sets": True,
    }:
        raise ValueError("Unsupported episode sampling protocol")
    controls = payload["training_control"]
    protocol = EpisodeProtocol(
        way=int(payload["way"]),
        support_shots=tuple(sorted(int(item) for item in payload["support_shots"])),
        query_patients_per_class=int(payload["query_patients_per_class"]),
        episode_counts={
            partition: int(payload["episode_counts"][partition])
            for partition in PARTITIONS
        },
        master_seed=int(payload["master_seed"]),
        validation_interval_episodes=int(
            controls["validation_interval_episodes"]
        ),
        minimum_train_episodes=int(controls["minimum_train_episodes"]),
        early_stopping_patience_checks=int(
            controls["early_stopping_patience_checks"]
        ),
        maximum_train_episodes=int(controls["maximum_train_episodes"]),
    )
    if protocol.way < 2:
        raise ValueError("way must be at least two")
    if not protocol.support_shots or protocol.support_shots[0] < 1:
        raise ValueError("support shots must be positive")
    if len(set(protocol.support_shots)) != len(protocol.support_shots):
        raise ValueError("support shots must be unique")
    if protocol.query_patients_per_class < 1:
        raise ValueError("query patient count must be positive")
    if any(protocol.episode_counts[item] < 1 for item in PARTITIONS):
        raise ValueError("episode counts must be positive")
    if protocol.maximum_train_episodes != protocol.episode_counts["train"]:
        raise ValueError("maximum train episodes must match the train episode count")
    if protocol.minimum_train_episodes > protocol.maximum_train_episodes:
        raise ValueError("minimum train episodes cannot exceed the maximum")
    return protocol


def load_split_image_manifest(
    path: Path,
) -> dict[tuple[str, str], dict[str, ClassPool]]:
    """Load the expanded split manifest into patient-indexed class pools."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(SplitImage.__dataclass_fields__)
    if not rows:
        raise ValueError("Split image manifest contains no rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Split image manifest is missing: " + ", ".join(sorted(missing)))

    images = [
        SplitImage(**{field: row[field] for field in SplitImage.__dataclass_fields__})
        for row in rows
    ]
    unique_keys = [(item.split_id, item.image_id) for item in images]
    if len(unique_keys) != len(set(unique_keys)):
        raise ValueError("Each image must occur once per split")
    if any(item.partition not in PARTITIONS for item in images):
        raise ValueError("Unknown partition in split image manifest")

    grouped: dict[
        tuple[str, str, str], dict[str, list[SplitImage]]
    ] = defaultdict(lambda: defaultdict(list))
    metadata: dict[tuple[str, str, str], tuple[str, str]] = {}
    patient_classes: dict[tuple[str, str], str] = {}
    class_partitions: dict[tuple[str, str], str] = {}
    for image in images:
        group_key = (image.split_id, image.partition, image.class_id)
        expected = metadata.setdefault(group_key, (image.class_name, image.category))
        if expected != (image.class_name, image.category):
            raise ValueError(f"Inconsistent metadata for {group_key}")
        patient_key = (image.split_id, image.patient_id)
        previous_class = patient_classes.setdefault(patient_key, image.class_id)
        if previous_class != image.class_id:
            raise ValueError(f"Patient {image.patient_id} occurs in multiple classes")
        class_key = (image.split_id, image.class_id)
        previous_partition = class_partitions.setdefault(class_key, image.partition)
        if previous_partition != image.partition:
            raise ValueError(
                f"Class {image.class_id} occurs in multiple partitions of {image.split_id}"
            )
        grouped[group_key][image.patient_id].append(image)

    pools: dict[tuple[str, str], dict[str, ClassPool]] = defaultdict(dict)
    for (split_id, partition, class_id), patients in sorted(grouped.items()):
        class_name, category = metadata[(split_id, partition, class_id)]
        pools[(split_id, partition)][class_id] = ClassPool(
            class_id=class_id,
            class_name=class_name,
            category=category,
            images_by_patient={
                patient_id: tuple(
                    sorted(members, key=lambda item: item.source_relpath.casefold())
                )
                for patient_id, members in sorted(patients.items())
            },
        )
    return dict(pools)


def _derived_seed(master_seed: int, *parts: object) -> int:
    material = "|".join([str(master_seed), *(str(item) for item in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _balanced_partial_cycle(
    combinations: Sequence[tuple[str, ...]],
    class_ids: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    """Choose a partial combination cycle with near-equal class exposure."""

    if count == 0:
        return []
    available = list(combinations)
    rng.shuffle(available)
    selected: list[tuple[str, ...]] = []
    exposure: Counter[str] = Counter()
    for _ in range(count):
        best_score: tuple[int, int] | None = None
        best_indices: list[int] = []
        for index, combination in enumerate(available):
            projected = exposure.copy()
            projected.update(combination)
            values = [projected[class_id] for class_id in class_ids]
            score = (max(values) - min(values), sum(value * value for value in values))
            if best_score is None or score < best_score:
                best_score = score
                best_indices = [index]
            elif score == best_score:
                best_indices.append(index)
        chosen_index = rng.choice(best_indices)
        chosen = available.pop(chosen_index)
        selected.append(chosen)
        exposure.update(chosen)

    def score(counter: Counter[str]) -> tuple[int, int]:
        values = [counter[class_id] for class_id in class_ids]
        return max(values) - min(values), sum(value * value for value in values)

    while True:
        current_score = score(exposure)
        if current_score[0] <= 1:
            break
        best_score = current_score
        best_swaps: list[tuple[int, int, Counter[str]]] = []
        for selected_index, old_combination in enumerate(selected):
            for available_index, new_combination in enumerate(available):
                projected = exposure.copy()
                projected.subtract(old_combination)
                projected.update(new_combination)
                projected_score = score(projected)
                if projected_score < best_score:
                    best_score = projected_score
                    best_swaps = [(selected_index, available_index, projected)]
                elif projected_score == best_score and projected_score < current_score:
                    best_swaps.append((selected_index, available_index, projected))
        if not best_swaps:
            break
        selected_index, available_index, exposure = rng.choice(best_swaps)
        selected[selected_index], available[available_index] = (
            available[available_index],
            selected[selected_index],
        )
    return selected


def balanced_class_schedule(
    class_ids: Sequence[str],
    way: int,
    episode_count: int,
    seed: int,
) -> list[tuple[str, ...]]:
    """Return a deterministic schedule balanced over all class combinations."""

    class_ids = tuple(sorted(class_ids))
    if len(class_ids) < way:
        raise ValueError(f"Cannot draw {way}-way episodes from {len(class_ids)} classes")
    combinations = list(itertools.combinations(class_ids, way))
    full_cycles, remainder = divmod(episode_count, len(combinations))
    rng = random.Random(seed)
    schedule: list[tuple[str, ...]] = []
    for _ in range(full_cycles):
        cycle = combinations.copy()
        rng.shuffle(cycle)
        schedule.extend(cycle)
    schedule.extend(
        _balanced_partial_cycle(combinations, class_ids, remainder, rng)
    )
    return schedule


def _sample_class(
    pool: ClassPool,
    protocol: EpisodeProtocol,
    rng: random.Random,
) -> dict[str, object]:
    required = protocol.maximum_shot + protocol.query_patients_per_class
    patient_ids = sorted(pool.images_by_patient)
    if len(patient_ids) < required:
        raise ValueError(
            f"Class {pool.class_id} has {len(patient_ids)} patients; {required} required"
        )
    selected_patients = rng.sample(patient_ids, required)
    support_patients = selected_patients[: protocol.maximum_shot]
    query_patients = selected_patients[protocol.maximum_shot :]

    def selected_image(patient_id: str) -> dict[str, str]:
        image = rng.choice(pool.images_by_patient[patient_id])
        return {
            "patient_id": patient_id,
            "image_id": image.image_id,
            "source_relpath": image.source_relpath,
        }

    support = []
    for rank, patient_id in enumerate(support_patients, start=1):
        item: dict[str, object] = selected_image(patient_id)
        item["support_rank"] = rank
        item["used_in_shots"] = [
            shot for shot in protocol.support_shots if rank <= shot
        ]
        support.append(item)
    query = []
    for rank, patient_id in enumerate(query_patients, start=1):
        item = selected_image(patient_id)
        item["query_rank"] = rank
        query.append(item)
    return {
        "class_id": pool.class_id,
        "class_name": pool.class_name,
        "category": pool.category,
        "support": support,
        "query": query,
    }


def iter_episodes(
    split_id: str,
    partition: str,
    pools: dict[str, ClassPool],
    protocol: EpisodeProtocol,
) -> Iterator[dict[str, object]]:
    """Yield deterministic episodes; 3-shot support is nested inside 5-shot."""

    schedule_seed = _derived_seed(
        protocol.master_seed, split_id, partition, "class_schedule"
    )
    schedule = balanced_class_schedule(
        sorted(pools),
        protocol.way,
        protocol.episode_counts[partition],
        schedule_seed,
    )
    for episode_index, class_ids in enumerate(schedule, start=1):
        episode_seed = _derived_seed(
            protocol.master_seed, split_id, partition, episode_index
        )
        rng = random.Random(episode_seed)
        classes = [
            {"class_slot": slot, **_sample_class(pools[class_id], protocol, rng)}
            for slot, class_id in enumerate(class_ids, start=1)
        ]
        yield {
            "episode_id": f"{split_id}:{partition}:{episode_index:06d}",
            "split_id": split_id,
            "partition": partition,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "way": protocol.way,
            "support_shots": list(protocol.support_shots),
            "query_patients_per_class": protocol.query_patients_per_class,
            "class_ids": list(class_ids),
            "classes": classes,
        }


def _write_jsonl_gz(path: Path, episodes: Iterator[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=raw_handle,
            mtime=0,
        ) as compressed_handle:
            with io.TextIOWrapper(
                compressed_handle, encoding="utf-8", newline="\n"
            ) as handle:
                for episode in episodes:
                    handle.write(
                        json.dumps(
                            episode,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")
                    count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_episode(
    episode: dict[str, object],
    pools: dict[str, ClassPool],
    protocol: EpisodeProtocol,
) -> None:
    class_ids = episode["class_ids"]
    classes = episode["classes"]
    if len(class_ids) != protocol.way or len(set(class_ids)) != protocol.way:
        raise ValueError(f"Invalid class selection in {episode['episode_id']}")
    if len(classes) != protocol.way:
        raise ValueError(f"Invalid class payload in {episode['episode_id']}")
    for class_item in classes:
        class_id = class_item["class_id"]
        if class_id not in pools:
            raise ValueError(f"Class {class_id} is outside the episode partition")
        support = class_item["support"]
        query = class_item["query"]
        if len(support) != protocol.maximum_shot:
            raise ValueError(f"Invalid support size in {episode['episode_id']}")
        if len(query) != protocol.query_patients_per_class:
            raise ValueError(f"Invalid query size in {episode['episode_id']}")
        support_patients = {item["patient_id"] for item in support}
        query_patients = {item["patient_id"] for item in query}
        if len(support_patients) != len(support) or len(query_patients) != len(query):
            raise ValueError(f"Repeated patient in {episode['episode_id']}")
        if support_patients & query_patients:
            raise ValueError(f"Support-query patient leakage in {episode['episode_id']}")
        for item in [*support, *query]:
            patient_id = item["patient_id"]
            valid_images = {
                image.image_id for image in pools[class_id].images_by_patient[patient_id]
            }
            if item["image_id"] not in valid_images:
                raise ValueError(f"Image-patient mismatch in {episode['episode_id']}")
        for item in support:
            rank = item["support_rank"]
            expected_shots = [
                shot for shot in protocol.support_shots if rank <= shot
            ]
            if item["used_in_shots"] != expected_shots:
                raise ValueError(f"Non-nested support set in {episode['episode_id']}")


def read_episode_manifest(path: Path) -> Iterator[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def generate_episode_manifests(
    pools: dict[tuple[str, str], dict[str, ClassPool]],
    protocol: EpisodeProtocol,
    output_dir: Path,
    source_manifest: Path,
    protocol_path: Path,
) -> dict[str, object]:
    """Materialise fixed compressed manifests and analysis summaries."""

    split_ids = sorted({split_id for split_id, _ in pools})
    missing = [
        f"{split_id}/{partition}"
        for split_id in split_ids
        for partition in PARTITIONS
        if (split_id, partition) not in pools
    ]
    if missing:
        raise ValueError("Missing split partitions: " + ", ".join(missing))

    required_patients = protocol.maximum_shot + protocol.query_patients_per_class
    insufficient = []
    for key, class_pools in pools.items():
        for class_id, pool in class_pools.items():
            if len(pool.images_by_patient) < required_patients:
                insufficient.append(
                    f"{key[0]}/{key[1]}/{class_id}={len(pool.images_by_patient)}"
                )
    if insufficient:
        raise ValueError(
            f"Classes below the {required_patients}-patient episode requirement: "
            + ", ".join(insufficient)
        )

    manifest_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    for split_id in split_ids:
        for partition in PARTITIONS:
            class_pools = pools[(split_id, partition)]
            output_path = output_dir / split_id / f"{partition}_episodes.jsonl.gz"
            episodes = iter_episodes(split_id, partition, class_pools, protocol)
            episode_count = _write_jsonl_gz(output_path, episodes)

            combination_counts: Counter[tuple[str, ...]] = Counter()
            class_exposure: Counter[str] = Counter()
            for episode in read_episode_manifest(output_path):
                validate_episode(episode, class_pools, protocol)
                class_ids = tuple(episode["class_ids"])
                combination_counts[class_ids] += 1
                class_exposure.update(class_ids)

            relative_path = output_path.relative_to(output_dir).as_posix()
            manifest_rows.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "episode_count": episode_count,
                    "class_pool_count": len(class_pools),
                    "class_combination_count": len(combination_counts),
                    "minimum_class_exposure": min(class_exposure.values()),
                    "maximum_class_exposure": max(class_exposure.values()),
                    "selected_patient_slots": (
                        episode_count * protocol.way * required_patients
                    ),
                    "manifest_path": relative_path,
                    "manifest_size_bytes": output_path.stat().st_size,
                    "manifest_sha256": _sha256(output_path),
                }
            )
            for combination, count in sorted(combination_counts.items()):
                combination_rows.append(
                    {
                        "split_id": split_id,
                        "partition": partition,
                        "class_combination": "|".join(combination),
                        "episode_count": count,
                    }
                )
            for class_id, count in sorted(class_exposure.items()):
                pool = class_pools[class_id]
                exposure_rows.append(
                    {
                        "split_id": split_id,
                        "partition": partition,
                        "class_id": class_id,
                        "class_name": pool.class_name,
                        "patient_count": len(pool.images_by_patient),
                        "episode_count": count,
                        "maximum_shot_support_slots": count * protocol.maximum_shot,
                        "query_slots": count * protocol.query_patients_per_class,
                    }
                )

    _write_csv(output_dir / "episode_manifest_index.csv", manifest_rows)
    _write_csv(output_dir / "class_combination_summary.csv", combination_rows)
    _write_csv(output_dir / "class_exposure_summary.csv", exposure_rows)
    summary = {
        "schema_version": 1,
        "source_split_manifest": str(source_manifest.resolve()),
        "source_split_manifest_sha256": _sha256(source_manifest),
        "episode_protocol": str(protocol_path.resolve()),
        "episode_protocol_sha256": _sha256(protocol_path),
        "split_count": len(split_ids),
        "way": protocol.way,
        "support_shots": list(protocol.support_shots),
        "query_patients_per_class": protocol.query_patients_per_class,
        "episode_counts_per_split": protocol.episode_counts,
        "total_episode_count": sum(
            int(row["episode_count"]) for row in manifest_rows
        ),
        "master_seed": protocol.master_seed,
        "nested_support_sets": True,
        "training_control": {
            "validation_interval_episodes": protocol.validation_interval_episodes,
            "minimum_train_episodes": protocol.minimum_train_episodes,
            "early_stopping_patience_checks": protocol.early_stopping_patience_checks,
            "maximum_train_episodes": protocol.maximum_train_episodes,
        },
        "manifests": manifest_rows,
    }
    (output_dir / "episode_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/class_splits/split_image_manifest.csv"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/episode_protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/episodes"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_episode_protocol(args.protocol)
    pools = load_split_image_manifest(args.split_manifest)
    summary = generate_episode_manifests(
        pools,
        protocol,
        args.output_dir,
        args.split_manifest,
        args.protocol,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
