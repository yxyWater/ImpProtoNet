"""Generate fixed-query, repeated-support manifests for experiment B."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .episodes import (
    ClassPool,
    EpisodeProtocol,
    SplitImage,
    load_episode_protocol,
    load_split_image_manifest,
    validate_episode,
)


@dataclass(frozen=True)
class SupportSensitivityProtocol:
    experiment_id: str
    partitions: tuple[str, ...]
    way: int
    support_shots: tuple[int, ...]
    query_patients_per_class: int
    query_panels_per_combination: int
    maximum_support_resamples: int
    pilot_prefixes: tuple[int, ...]
    pilot_split_id: str
    pilot_model_ids: tuple[str, ...]
    convergence_absolute_change_threshold: float
    consecutive_stable_prefixes: int
    fallback_resamples: int
    master_seed: int

    @property
    def maximum_shot(self) -> int:
        return max(self.support_shots)


def load_support_sensitivity_protocol(
    path: Path,
    episode_protocol_path: Path,
) -> SupportSensitivityProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = load_episode_protocol(episode_protocol_path)
    if payload.get("schema_version") != 1 or payload.get("experiment_id") != "B":
        raise ValueError("Unsupported support-sensitivity configuration")
    if payload.get("partitions") != ["validation", "test"]:
        raise ValueError("Experiment B must generate validation and test manifests")
    if payload.get("class_combination_policy") != "all_combinations":
        raise ValueError("Experiment B must cover every test-class combination")
    if int(payload.get("query_panels_per_combination", 0)) != 1:
        raise ValueError("Exactly one fixed query panel per combination is approved")
    if int(payload.get("query_patients_per_class", 0)) != base.query_patients_per_class:
        raise ValueError("Experiment B query count must match experiment A")
    shots = tuple(int(value) for value in payload.get("support_shots", ()))
    if shots != base.support_shots:
        raise ValueError("Experiment B shots must match experiment A")
    if int(payload.get("maximum_support_resamples", 0)) != 100:
        raise ValueError("Experiment B must pre-generate 100 support resamples")
    prefixes = tuple(int(value) for value in payload.get("pilot_prefixes", ()))
    if prefixes != (20, 30, 40, 50, 75, 100):
        raise ValueError("Experiment B pilot prefixes differ from the confirmed plan")
    expected_sampling = {
        "patient_unit": True,
        "fixed_query_patient_and_image": True,
        "fixed_image_per_patient_across_resamples": True,
        "query_support_patient_disjoint": True,
        "nested_support_sets": True,
        "unique_complete_5shot_support_sets": True,
        "unique_nested_3shot_support_sets": True,
    }
    if payload.get("sampling") != expected_sampling:
        raise ValueError("Experiment B sampling safeguards differ from the confirmed plan")
    pilot = payload["pilot"]
    if pilot.get("split_id") != "split_01":
        raise ValueError("Experiment B convergence pilot is restricted to split_01")
    if pilot.get("model_ids") != ["M0", "M3", "M5"]:
        raise ValueError("Experiment B pilot must use M0, M3, and M5")
    if float(pilot.get("absolute_change_threshold", -1)) != 0.01:
        raise ValueError("Experiment B convergence threshold must be 0.01")
    if int(pilot.get("consecutive_stable_prefixes", 0)) != 2:
        raise ValueError("Experiment B requires two consecutive stable prefixes")
    if int(pilot.get("fallback_resamples", 0)) != 100:
        raise ValueError("Experiment B fallback must use all 100 resamples")
    return SupportSensitivityProtocol(
        experiment_id="B",
        partitions=("validation", "test"),
        way=base.way,
        support_shots=shots,
        query_patients_per_class=base.query_patients_per_class,
        query_panels_per_combination=1,
        maximum_support_resamples=100,
        pilot_prefixes=prefixes,
        pilot_split_id="split_01",
        pilot_model_ids=("M0", "M3", "M5"),
        convergence_absolute_change_threshold=0.01,
        consecutive_stable_prefixes=2,
        fallback_resamples=100,
        master_seed=int(payload["master_seed"]),
    )


def _derived_seed(master_seed: int, *parts: object) -> int:
    material = "|".join([str(master_seed), *(str(item) for item in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _fixed_image(
    pool: ClassPool,
    patient_id: str,
    protocol: SupportSensitivityProtocol,
    split_id: str,
    partition: str,
) -> SplitImage:
    images = pool.images_by_patient[patient_id]
    seed = _derived_seed(
        protocol.master_seed,
        split_id,
        partition,
        pool.class_id,
        patient_id,
        "fixed_image",
    )
    return images[seed % len(images)]


def _image_item(image: SplitImage, patient_id: str) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "image_id": image.image_id,
        "source_relpath": image.source_relpath,
    }


def _validation_protocol(protocol: SupportSensitivityProtocol) -> EpisodeProtocol:
    return EpisodeProtocol(
        way=protocol.way,
        support_shots=protocol.support_shots,
        query_patients_per_class=protocol.query_patients_per_class,
        episode_counts={"train": 1, "validation": 1, "test": 1},
        master_seed=protocol.master_seed,
        validation_interval_episodes=1,
        minimum_train_episodes=1,
        early_stopping_patience_checks=1,
        maximum_train_episodes=1,
    )


def iter_support_sensitivity_episodes(
    split_id: str,
    partition: str,
    pools: dict[str, ClassPool],
    protocol: SupportSensitivityProtocol,
) -> Iterator[dict[str, object]]:
    """Yield every 3-way combination with one fixed query panel and 100 supports."""

    if partition not in protocol.partitions:
        raise ValueError("Experiment B only supports validation and test partitions")
    class_ids = tuple(sorted(pools))
    if len(class_ids) != 5:
        raise ValueError("Experiment B requires exactly five classes per partition")
    combinations = tuple(itertools.combinations(class_ids, protocol.way))
    validator = _validation_protocol(protocol)
    episode_index = 0
    for combination_index, combination in enumerate(combinations, start=1):
        query_by_class: dict[str, tuple[str, ...]] = {}
        candidates_by_class: dict[str, tuple[str, ...]] = {}
        for class_id in combination:
            pool = pools[class_id]
            patients = tuple(sorted(pool.images_by_patient))
            required = protocol.query_patients_per_class + protocol.maximum_shot
            if len(patients) < required:
                raise ValueError(
                    f"Class {class_id} has {len(patients)} patients; {required} required"
                )
            rng = random.Random(
                _derived_seed(
                    protocol.master_seed,
                    split_id,
                    partition,
                    combination_index,
                    class_id,
                    "query_panel",
                )
            )
            query_patients = tuple(
                rng.sample(list(patients), protocol.query_patients_per_class)
            )
            query_by_class[class_id] = query_patients
            query_set = set(query_patients)
            candidates_by_class[class_id] = tuple(
                patient_id for patient_id in patients if patient_id not in query_set
            )
        possible_complete_sets = math.prod(
            math.comb(len(candidates_by_class[class_id]), protocol.maximum_shot)
            for class_id in combination
        )
        possible_nested_sets = math.prod(
            math.comb(len(candidates_by_class[class_id]), min(protocol.support_shots))
            for class_id in combination
        )
        if min(possible_complete_sets, possible_nested_sets) < protocol.maximum_support_resamples:
            raise ValueError(
                f"Combination {combination} cannot provide 100 unique nested support sets"
            )

        seen_complete: set[tuple[tuple[str, ...], ...]] = set()
        seen_nested: set[tuple[tuple[str, ...], ...]] = set()
        for resample_index in range(1, protocol.maximum_support_resamples + 1):
            selected_by_class: dict[str, tuple[str, ...]] | None = None
            for attempt in range(10000):
                proposed: dict[str, tuple[str, ...]] = {}
                for class_id in combination:
                    rng = random.Random(
                        _derived_seed(
                            protocol.master_seed,
                            split_id,
                            partition,
                            combination_index,
                            resample_index,
                            attempt,
                            class_id,
                            "support",
                        )
                    )
                    proposed[class_id] = tuple(
                        rng.sample(
                            list(candidates_by_class[class_id]),
                            protocol.maximum_shot,
                        )
                    )
                complete_signature = tuple(
                    tuple(sorted(proposed[class_id])) for class_id in combination
                )
                nested_signature = tuple(
                    tuple(sorted(proposed[class_id][: min(protocol.support_shots)]))
                    for class_id in combination
                )
                if complete_signature in seen_complete or nested_signature in seen_nested:
                    continue
                seen_complete.add(complete_signature)
                seen_nested.add(nested_signature)
                selected_by_class = proposed
                break
            if selected_by_class is None:
                raise RuntimeError(
                    f"Could not construct unique support resample {resample_index} for {combination}"
                )

            classes = []
            for class_slot, class_id in enumerate(combination, start=1):
                pool = pools[class_id]
                support = []
                for rank, patient_id in enumerate(selected_by_class[class_id], start=1):
                    item = _image_item(
                        _fixed_image(pool, patient_id, protocol, split_id, partition),
                        patient_id,
                    )
                    item["support_rank"] = rank
                    item["used_in_shots"] = [
                        shot for shot in protocol.support_shots if rank <= shot
                    ]
                    support.append(item)
                query = []
                for rank, patient_id in enumerate(query_by_class[class_id], start=1):
                    item = _image_item(
                        _fixed_image(pool, patient_id, protocol, split_id, partition),
                        patient_id,
                    )
                    item["query_rank"] = rank
                    query.append(item)
                classes.append(
                    {
                        "class_slot": class_slot,
                        "class_id": class_id,
                        "class_name": pool.class_name,
                        "category": pool.category,
                        "support": support,
                        "query": query,
                    }
                )
            episode_index += 1
            episode_seed = _derived_seed(
                protocol.master_seed,
                split_id,
                partition,
                combination_index,
                resample_index,
            )
            episode = {
                "episode_id": (
                    f"{split_id}:{partition}:B:c{combination_index:02d}:"
                    f"r{resample_index:03d}"
                ),
                "experiment_id": protocol.experiment_id,
                "split_id": split_id,
                "partition": partition,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "way": protocol.way,
                "support_shots": list(protocol.support_shots),
                "query_patients_per_class": protocol.query_patients_per_class,
                "class_ids": list(combination),
                "disease_combination_index": combination_index,
                "query_panel_index": 1,
                "support_resample_index": resample_index,
                "maximum_support_resamples": protocol.maximum_support_resamples,
                "fixed_query_panel": True,
                "fixed_image_per_patient": True,
                "classes": classes,
            }
            validate_episode(episode, pools, validator)
            yield episode


def _write_jsonl_gz(path: Path, episodes: Iterator[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw_handle, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for episode in episodes:
                    handle.write(
                        json.dumps(episode, ensure_ascii=False, separators=(",", ":"))
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def generate_support_sensitivity_manifests(
    split_manifest_path: Path,
    episode_protocol_path: Path,
    support_protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    protocol = load_support_sensitivity_protocol(
        support_protocol_path, episode_protocol_path
    )
    all_pools = load_split_image_manifest(split_manifest_path)
    index_rows = []
    for split_number in range(1, 6):
        split_id = f"split_{split_number:02d}"
        for partition in protocol.partitions:
            pools = all_pools[(split_id, partition)]
            path = output_dir / split_id / f"{partition}_episodes.jsonl.gz"
            count = _write_jsonl_gz(
                path,
                iter_support_sensitivity_episodes(
                    split_id, partition, pools, protocol
                ),
            )
            expected = math.comb(len(pools), protocol.way) * protocol.maximum_support_resamples
            if count != expected:
                raise RuntimeError(
                    f"{split_id} {partition} wrote {count} episodes; {expected} expected"
                )
            index_rows.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "episode_count": count,
                    "manifest_path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
    _write_csv(output_dir / "manifest_index.csv", index_rows)
    summary = {
        "schema_version": 1,
        "experiment_id": protocol.experiment_id,
        "protocol": asdict(protocol),
        "split_manifest_path": str(split_manifest_path.resolve()),
        "split_manifest_sha256": _sha256(split_manifest_path),
        "episode_protocol_path": str(episode_protocol_path.resolve()),
        "episode_protocol_sha256": _sha256(episode_protocol_path),
        "support_protocol_path": str(support_protocol_path.resolve()),
        "support_protocol_sha256": _sha256(support_protocol_path),
        "manifest_count": len(index_rows),
        "total_episode_count": sum(int(row["episode_count"]) for row in index_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/class_splits/split_image_manifest.csv"),
    )
    parser.add_argument(
        "--episode-protocol",
        type=Path,
        default=Path("configs/episode_protocol.json"),
    )
    parser.add_argument(
        "--support-protocol",
        type=Path,
        default=Path("configs/support_sensitivity.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/support_sensitivity_episodes"),
    )
    args = parser.parse_args(argv)
    summary = generate_support_sensitivity_manifests(
        args.split_manifest,
        args.episode_protocol,
        args.support_protocol,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

