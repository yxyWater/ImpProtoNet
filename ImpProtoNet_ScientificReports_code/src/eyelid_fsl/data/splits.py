"""Generate deterministic, class-disjoint few-shot dataset splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ManifestImage:
    image_id: str
    class_id: str
    class_name: str
    category: str
    patient_id: str
    source_relpath: str


@dataclass(frozen=True)
class ClassStatistics:
    class_id: str
    class_name: str
    category: str
    patient_count: int
    image_count: int


@dataclass(frozen=True)
class ClassSplit:
    split_id: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    candidate_count: int
    balance_score: float
    splits: tuple[ClassSplit, ...]


def _parse_bool(value: str) -> bool:
    normalised = value.strip().casefold()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value in manifest: {value!r}")


def load_included_manifest(path: Path) -> list[ManifestImage]:
    """Load only audit-approved images and validate patient/class consistency."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "class_id",
        "class_name",
        "category",
        "patient_id",
        "source_relpath",
        "include_in_experiments",
    }
    if not raw_rows:
        raise ValueError("Patient manifest contains no rows")
    missing = required.difference(raw_rows[0])
    if missing:
        raise ValueError("Patient manifest is missing columns: " + ", ".join(sorted(missing)))

    records = [
        ManifestImage(
            image_id=row["image_id"],
            class_id=row["class_id"],
            class_name=row["class_name"],
            category=row["category"],
            patient_id=row["patient_id"],
            source_relpath=row["source_relpath"],
        )
        for row in raw_rows
        if _parse_bool(row["include_in_experiments"])
    ]
    if not records:
        raise ValueError("Patient manifest contains no experiment-approved images")

    image_ids = [record.image_id for record in records]
    source_paths = [record.source_relpath for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Included image IDs must be unique")
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("Included source paths must be unique")

    patient_classes: dict[str, str] = {}
    class_metadata: dict[str, tuple[str, str]] = {}
    for record in records:
        previous_class = patient_classes.setdefault(record.patient_id, record.class_id)
        if previous_class != record.class_id:
            raise ValueError(
                f"Patient {record.patient_id!r} occurs in multiple disease classes"
            )
        metadata = (record.class_name, record.category)
        previous_metadata = class_metadata.setdefault(record.class_id, metadata)
        if previous_metadata != metadata:
            raise ValueError(f"Inconsistent metadata for class {record.class_id}")
    return sorted(records, key=lambda item: (item.class_id, item.source_relpath.casefold()))


def build_class_statistics(
    records: Sequence[ManifestImage],
    min_patients: int = 15,
) -> list[ClassStatistics]:
    """Aggregate included images by class and enforce the experiment threshold."""

    by_class: dict[str, list[ManifestImage]] = defaultdict(list)
    for record in records:
        by_class[record.class_id].append(record)

    statistics: list[ClassStatistics] = []
    insufficient: list[str] = []
    for class_id, members in sorted(by_class.items()):
        patients = {member.patient_id for member in members}
        stat = ClassStatistics(
            class_id=class_id,
            class_name=members[0].class_name,
            category=members[0].category,
            patient_count=len(patients),
            image_count=len(members),
        )
        statistics.append(stat)
        if stat.patient_count < min_patients:
            insufficient.append(f"{class_id}({stat.patient_count})")
    if insufficient:
        raise ValueError(
            f"Classes below the {min_patients}-patient threshold: "
            + ", ".join(insufficient)
        )
    return statistics


def _assign_benign_classes(
    fold_sets: list[set[str]],
    benign_ids: Sequence[str],
    repeated_ids: set[str],
    forbidden_by_fold: Sequence[set[str]],
    rng: random.Random,
) -> bool:
    remaining = set(benign_ids)
    occurrence_counts = {
        class_id: 2 if class_id in repeated_ids else 1 for class_id in benign_ids
    }

    def feasible_combinations(class_id: str) -> list[tuple[int, ...]]:
        count = occurrence_counts[class_id]
        eligible = [
            fold
            for fold in range(len(fold_sets))
            if len(fold_sets[fold]) < 5
            and class_id not in fold_sets[fold]
            and class_id not in forbidden_by_fold[fold]
        ]
        combinations = list(itertools.combinations(eligible, count))
        rng.shuffle(combinations)
        combinations.sort(
            key=lambda combo: (
                sum((len(fold_sets[fold]) + 1) ** 2 for fold in combo),
                max((len(fold_sets[fold]) + 1 for fold in combo), default=0),
            )
        )
        return combinations

    def backtrack() -> bool:
        if not remaining:
            return all(len(fold) == 5 for fold in fold_sets)

        candidates: list[tuple[int, int, float, str, list[tuple[int, ...]]]] = []
        for class_id in sorted(remaining):
            combinations = feasible_combinations(class_id)
            if not combinations:
                return False
            candidates.append(
                (
                    len(combinations),
                    -occurrence_counts[class_id],
                    rng.random(),
                    class_id,
                    combinations,
                )
            )
        _, _, _, class_id, combinations = min(candidates)
        remaining.remove(class_id)
        for combo in combinations:
            for fold in combo:
                fold_sets[fold].add(class_id)
            if backtrack():
                return True
            for fold in combo:
                fold_sets[fold].remove(class_id)
        remaining.add(class_id)
        return False

    return backtrack()


def _build_role_schedule(
    malignant_ids: Sequence[str],
    benign_ids: Sequence[str],
    repeated_ids: set[str],
    forbidden_by_fold: Sequence[set[str]],
    rng: random.Random,
) -> list[set[str]] | None:
    fold_count = len(malignant_ids)
    permutations = [
        permutation
        for permutation in itertools.permutations(malignant_ids)
        if all(
            class_id not in forbidden_by_fold[fold]
            for fold, class_id in enumerate(permutation)
        )
    ]
    rng.shuffle(permutations)
    for permutation in permutations:
        fold_sets = [{class_id} for class_id in permutation]
        if _assign_benign_classes(
            fold_sets,
            benign_ids,
            repeated_ids,
            forbidden_by_fold,
            rng,
        ):
            return fold_sets
    return None


def _normalised_variance(values: Iterable[int]) -> float:
    values = list(values)
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return sum(((value - mean) / mean) ** 2 for value in values) / len(values)


def _balance_score(
    splits: Sequence[ClassSplit],
    statistics: dict[str, ClassStatistics],
) -> float:
    score = 0.0
    for partition in ("train", "validation", "test"):
        patient_totals = []
        image_totals = []
        for split in splits:
            class_ids = getattr(split, partition)
            patient_totals.append(sum(statistics[item].patient_count for item in class_ids))
            image_totals.append(sum(statistics[item].image_count for item in class_ids))
        score += _normalised_variance(patient_totals)
        score += 0.25 * _normalised_variance(image_totals)
    return score


def validate_split_plan(
    plan: SplitPlan,
    statistics: Sequence[ClassStatistics],
    train_count: int = 12,
    validation_count: int = 5,
    test_count: int = 5,
    malignant_category: str = "恶性",
) -> None:
    """Raise an error if a generated plan violates any frozen split rule."""

    all_ids = {item.class_id for item in statistics}
    malignant_ids = {
        item.class_id for item in statistics if item.category == malignant_category
    }
    validation_coverage: Counter[str] = Counter()
    test_coverage: Counter[str] = Counter()
    train_coverage: Counter[str] = Counter()

    if len(plan.splits) != 5:
        raise ValueError("Exactly five fixed splits are required")
    for split in plan.splits:
        partitions = {
            "train": set(split.train),
            "validation": set(split.validation),
            "test": set(split.test),
        }
        expected_counts = {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
        for name, class_ids in partitions.items():
            if len(class_ids) != expected_counts[name]:
                raise ValueError(f"{split.split_id} has an invalid {name} class count")
        if (
            partitions["train"] & partitions["validation"]
            or partitions["train"] & partitions["test"]
            or partitions["validation"] & partitions["test"]
        ):
            raise ValueError(f"{split.split_id} contains overlapping class partitions")
        if set.union(*partitions.values()) != all_ids:
            raise ValueError(f"{split.split_id} does not contain every class exactly once")
        if len(partitions["validation"] & malignant_ids) != 1:
            raise ValueError(f"{split.split_id} validation must contain one malignant class")
        if len(partitions["test"] & malignant_ids) != 1:
            raise ValueError(f"{split.split_id} test must contain one malignant class")
        if len(partitions["train"] & malignant_ids) != 3:
            raise ValueError(f"{split.split_id} train must contain three malignant classes")
        train_coverage.update(partitions["train"])
        validation_coverage.update(partitions["validation"])
        test_coverage.update(partitions["test"])

    if set(validation_coverage) != all_ids or set(test_coverage) != all_ids:
        raise ValueError("Every class must appear in validation and test at least once")
    if not all(count in {1, 2} for count in validation_coverage.values()):
        raise ValueError("Validation class coverage must be one or two occurrences")
    if not all(count in {1, 2} for count in test_coverage.values()):
        raise ValueError("Test class coverage must be one or two occurrences")
    validation_repeats = {
        class_id for class_id, count in validation_coverage.items() if count == 2
    }
    test_repeats = {
        class_id for class_id, count in test_coverage.items() if count == 2
    }
    if len(validation_repeats) != 3 or len(test_repeats) != 3:
        raise ValueError("Validation and test must each repeat exactly three classes")
    if validation_repeats & test_repeats:
        raise ValueError("Repeated validation and test classes must be disjoint")
    if validation_repeats & malignant_ids or test_repeats & malignant_ids:
        raise ValueError("Only benign classes may occupy the unavoidable repeat slots")
    if min(train_coverage.values()) < 2:
        raise ValueError("Every class must appear in training in at least two splits")


def generate_balanced_splits(
    statistics: Sequence[ClassStatistics],
    seed: int = 20261004,
    candidate_count: int = 2000,
    train_count: int = 12,
    validation_count: int = 5,
    test_count: int = 5,
    malignant_category: str = "恶性",
) -> SplitPlan:
    """Search deterministic valid candidates and retain the best balanced plan."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if train_count + validation_count + test_count != len(statistics):
        raise ValueError("Partition sizes must equal the number of disease classes")
    if validation_count != 5 or test_count != 5:
        raise ValueError("The approved design requires five validation and five test classes")

    by_id = {item.class_id: item for item in statistics}
    if len(by_id) != len(statistics):
        raise ValueError("Class IDs must be unique")
    malignant_ids = sorted(
        item.class_id for item in statistics if item.category == malignant_category
    )
    benign_ids = sorted(
        item.class_id for item in statistics if item.category != malignant_category
    )
    if len(malignant_ids) != 5 or len(benign_ids) != 17:
        raise ValueError(
            "The approved balanced design requires exactly 5 malignant and 17 benign classes"
        )

    rng = random.Random(seed)
    empty_forbidden = [set() for _ in range(5)]
    best_plan: SplitPlan | None = None
    best_signature: tuple[tuple[str, ...], ...] | None = None

    for _ in range(candidate_count):
        test_repeats = set(rng.sample(benign_ids, 3))
        validation_repeat_pool = [item for item in benign_ids if item not in test_repeats]
        validation_repeats = set(rng.sample(validation_repeat_pool, 3))

        test_folds = _build_role_schedule(
            malignant_ids,
            benign_ids,
            test_repeats,
            empty_forbidden,
            rng,
        )
        if test_folds is None:
            continue
        validation_folds = _build_role_schedule(
            malignant_ids,
            benign_ids,
            validation_repeats,
            test_folds,
            rng,
        )
        if validation_folds is None:
            continue

        all_ids = set(by_id)
        splits = tuple(
            ClassSplit(
                split_id=f"split_{fold + 1:02d}",
                train=tuple(sorted(all_ids - test_folds[fold] - validation_folds[fold])),
                validation=tuple(sorted(validation_folds[fold])),
                test=tuple(sorted(test_folds[fold])),
            )
            for fold in range(5)
        )
        score = _balance_score(splits, by_id)
        signature = tuple(
            item
            for split in splits
            for item in (split.train, split.validation, split.test)
        )
        candidate = SplitPlan(
            seed=seed,
            candidate_count=candidate_count,
            balance_score=score,
            splits=splits,
        )
        if (
            best_plan is None
            or score < best_plan.balance_score
            or (math.isclose(score, best_plan.balance_score) and signature < best_signature)
        ):
            validate_split_plan(
                candidate,
                statistics,
                train_count=train_count,
                validation_count=validation_count,
                test_count=test_count,
                malignant_category=malignant_category,
            )
            best_plan = candidate
            best_signature = signature

    if best_plan is None:
        raise RuntimeError("No valid class split plan could be generated")
    return best_plan


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split_outputs(
    output_dir: Path,
    plan: SplitPlan,
    statistics: Sequence[ClassStatistics],
    records: Sequence[ManifestImage],
    source_manifest: Path,
    min_patients: int,
) -> dict[str, object]:
    """Write frozen class assignments and expanded image-level manifests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {item.class_id: item for item in statistics}
    class_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    assignment_lookup: dict[tuple[str, str], str] = {}

    for split in plan.splits:
        for partition in ("train", "validation", "test"):
            class_ids = getattr(split, partition)
            partition_stats = [by_id[class_id] for class_id in class_ids]
            for stat in partition_stats:
                assignment_lookup[(split.split_id, stat.class_id)] = partition
                class_rows.append(
                    {
                        "split_id": split.split_id,
                        "partition": partition,
                        **asdict(stat),
                    }
                )
            summary_rows.append(
                {
                    "split_id": split.split_id,
                    "partition": partition,
                    "class_count": len(partition_stats),
                    "benign_class_count": sum(item.category != "恶性" for item in partition_stats),
                    "malignant_class_count": sum(item.category == "恶性" for item in partition_stats),
                    "patient_count": sum(item.patient_count for item in partition_stats),
                    "image_count": sum(item.image_count for item in partition_stats),
                }
            )

    for split in plan.splits:
        for record in records:
            image_rows.append(
                {
                    "split_id": split.split_id,
                    "partition": assignment_lookup[(split.split_id, record.class_id)],
                    **asdict(record),
                }
            )

    coverage_rows: list[dict[str, object]] = []
    for stat in statistics:
        counts = Counter(
            assignment_lookup[(split.split_id, stat.class_id)] for split in plan.splits
        )
        coverage_rows.append(
            {
                **asdict(stat),
                "train_split_count": counts["train"],
                "validation_split_count": counts["validation"],
                "test_split_count": counts["test"],
            }
        )

    _write_csv(
        output_dir / "class_assignments.csv",
        class_rows,
        ["split_id", "partition", *ClassStatistics.__dataclass_fields__],
    )
    _write_csv(
        output_dir / "split_image_manifest.csv",
        image_rows,
        ["split_id", "partition", *ManifestImage.__dataclass_fields__],
    )
    _write_csv(
        output_dir / "split_summary.csv",
        summary_rows,
        [
            "split_id",
            "partition",
            "class_count",
            "benign_class_count",
            "malignant_class_count",
            "patient_count",
            "image_count",
        ],
    )
    _write_csv(
        output_dir / "class_coverage.csv",
        coverage_rows,
        [
            *ClassStatistics.__dataclass_fields__,
            "train_split_count",
            "validation_split_count",
            "test_split_count",
        ],
    )

    payload = {
        "schema_version": 1,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _file_sha256(source_manifest),
        "included_image_count": len(records),
        "class_count": len(statistics),
        "seed": plan.seed,
        "candidate_count": plan.candidate_count,
        "balance_score": plan.balance_score,
        "constraints": {
            "split_count": 5,
            "train_class_count": 12,
            "validation_class_count": 5,
            "test_class_count": 5,
            "minimum_patients_per_class": min_patients,
            "validation_malignant_class_count": 1,
            "test_malignant_class_count": 1,
            "train_malignant_class_count": 3,
            "validation_and_test_class_disjoint": True,
            "all_partitions_class_disjoint": True,
            "repeat_slots_use_distinct_benign_classes": True,
        },
        "splits": [
            {
                "split_id": split.split_id,
                "train": list(split.train),
                "validation": list(split.validation),
                "test": list(split.test),
            }
            for split in plan.splits
        ],
    }
    (output_dir / "class_splits.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/dataset_audit/patient_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/class_splits"),
    )
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--candidate-count", type=int, default=2000)
    parser.add_argument("--min-patients", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_included_manifest(args.manifest)
    statistics = build_class_statistics(records, min_patients=args.min_patients)
    plan = generate_balanced_splits(
        statistics,
        seed=args.seed,
        candidate_count=args.candidate_count,
    )
    payload = write_split_outputs(
        args.output_dir,
        plan,
        statistics,
        records,
        args.manifest,
        args.min_patients,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
