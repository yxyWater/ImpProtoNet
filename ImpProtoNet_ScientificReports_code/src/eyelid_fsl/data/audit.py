"""Build a patient-level manifest from the original eyelid image collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LEADING_PATIENT_RE = re.compile(r"^([A-Za-z]{0,8}\d+(?:\.\d+)*)")
EMBEDDED_PATIENT_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,8}\d{3,}|\d{5,})(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class ClassSpec:
    class_id: str
    class_name: str
    category: str
    source_relpath: str


@dataclass(frozen=True)
class PatientIdentity:
    patient_id: str
    source: str
    raw_token: str | None


@dataclass(frozen=True)
class ExclusionRule:
    source_relpath: str
    reason: str
    canonical_source_relpath: str


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    class_id: str
    class_name: str
    category: str
    patient_id: str
    patient_id_source: str
    patient_id_raw_token: str
    source_relpath: str
    filename: str
    file_size_bytes: int
    sha256: str
    dhash64: str
    width: int
    height: int
    image_mode: str
    image_format: str
    include_in_experiments: bool
    exclusion_reason: str
    canonical_source_relpath: str


def _normalise_token(token: str) -> str:
    return token.upper()


def parse_patient_identity(filename: str, source_relpath: str) -> PatientIdentity:
    """Extract a patient ID or assign a unique per-file patient ID.

    The source convention groups names such as ``1262.jpg`` and ``1262-1.jpg``
    under patient 1262. Figure-style names may contain the patient number later in
    the stem. When no explicit number exists, the user-specified rule is that the
    image represents one independent patient.
    """

    stem = Path(filename).stem.strip()
    leading = LEADING_PATIENT_RE.match(stem)
    if leading:
        token = _normalise_token(leading.group(1))
        return PatientIdentity(f"PID:{token}", "filename_prefix", token)

    embedded = EMBEDDED_PATIENT_RE.findall(stem)
    if embedded:
        token = _normalise_token(embedded[-1])
        return PatientIdentity(f"PID:{token}", "embedded_code", token)

    normalised_path = source_relpath.replace("\\", "/")
    path_digest = hashlib.sha256(normalised_path.encode("utf-8")).hexdigest()[:20]
    return PatientIdentity(
        f"FILE:{path_digest}",
        "single_image_assumption",
        None,
    )


def _dhash64(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    get_pixels = getattr(grayscale, "get_flattened_data", grayscale.getdata)
    pixels = list(get_pixels())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:016x}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_record(
    path: Path,
    source_root: Path,
    class_spec: ClassSpec,
    exclusion_rules: dict[str, ExclusionRule],
) -> tuple[ImageRecord | None, dict[str, str] | None]:
    source_relpath = path.relative_to(source_root).as_posix()
    identity = parse_patient_identity(path.name, source_relpath)
    exclusion = exclusion_rules.get(source_relpath)
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_mode = image.mode
            image_format = image.format or path.suffix.lstrip(".").upper()
            perceptual_hash = _dhash64(image)
        file_hash = _sha256(path)
    except Exception as exc:  # Pillow exposes format-specific exception types.
        return None, {
            "class_id": class_spec.class_id,
            "class_name": class_spec.class_name,
            "source_relpath": source_relpath,
            "error": repr(exc),
        }

    image_id = hashlib.sha256(source_relpath.encode("utf-8")).hexdigest()[:20]
    return (
        ImageRecord(
            image_id=f"IMG:{image_id}",
            class_id=class_spec.class_id,
            class_name=class_spec.class_name,
            category=class_spec.category,
            patient_id=f"{class_spec.class_id}:{identity.patient_id}",
            patient_id_source=identity.source,
            patient_id_raw_token=identity.raw_token or "",
            source_relpath=source_relpath,
            filename=path.name,
            file_size_bytes=path.stat().st_size,
            sha256=file_hash,
            dhash64=perceptual_hash,
            width=width,
            height=height,
            image_mode=image_mode,
            image_format=image_format,
            include_in_experiments=exclusion is None,
            exclusion_reason=exclusion.reason if exclusion else "",
            canonical_source_relpath=(
                exclusion.canonical_source_relpath if exclusion else ""
            ),
        ),
        None,
    )


def load_class_specs(config_path: Path) -> list[ClassSpec]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    specs = [ClassSpec(**item) for item in payload["classes"]]
    ids = [item.class_id for item in specs]
    relpaths = [item.source_relpath for item in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Class IDs must be unique")
    if len(relpaths) != len(set(relpaths)):
        raise ValueError("Source class paths must be unique")
    return sorted(specs, key=lambda item: item.class_id)


def load_exclusion_rules(config_path: Path) -> dict[str, ExclusionRule]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    rules = [ExclusionRule(**item) for item in payload.get("exclusions", [])]
    paths = [rule.source_relpath for rule in rules]
    if len(paths) != len(set(paths)):
        raise ValueError("Excluded source paths must be unique")
    canonical_paths = {rule.canonical_source_relpath for rule in rules}
    overlap = canonical_paths.intersection(paths)
    if overlap:
        raise ValueError(
            "Canonical images cannot also be excluded: " + ", ".join(sorted(overlap))
        )
    return {rule.source_relpath: rule for rule in rules}


def _iter_class_images(source_root: Path, spec: ClassSpec) -> list[Path]:
    class_dir = source_root / Path(spec.source_relpath)
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Missing source class directory: {class_dir}")
    return sorted(
        (
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _duplicate_rows(records: Sequence[ImageRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_type, attribute in (("exact_sha256", "sha256"), ("equal_dhash64", "dhash64")):
        groups: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            groups[getattr(record, attribute)].append(record)
        group_index = 0
        for fingerprint, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            group_index += 1
            group_id = f"{group_type}:{group_index:04d}"
            for member in sorted(members, key=lambda item: item.source_relpath):
                rows.append(
                    {
                        "group_type": group_type,
                        "group_id": group_id,
                        "fingerprint": fingerprint,
                        "member_count": len(members),
                        "class_id": member.class_id,
                        "class_name": member.class_name,
                        "patient_id": member.patient_id,
                        "source_relpath": member.source_relpath,
                        "include_in_experiments": member.include_in_experiments,
                        "canonical_source_relpath": member.canonical_source_relpath,
                    }
                )
    return rows


def _class_summary_rows(
    specs: Sequence[ClassSpec],
    records: Sequence[ImageRecord],
    invalid_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_id].append(record)
    invalid_counts = Counter(row["class_id"] for row in invalid_rows)

    rows: list[dict[str, object]] = []
    for spec in specs:
        members = by_class[spec.class_id]
        included_members = [member for member in members if member.include_in_experiments]
        patient_counts = Counter(member.patient_id for member in included_members)
        patient_count = len(patient_counts)
        rows.append(
            {
                "class_id": spec.class_id,
                "class_name": spec.class_name,
                "category": spec.category,
                "source_relpath": spec.source_relpath,
                "valid_image_count": len(members),
                "included_image_count": len(included_members),
                "excluded_image_count": len(members) - len(included_members),
                "invalid_image_count": invalid_counts[spec.class_id],
                "patient_count": patient_count,
                "multi_image_patient_count": sum(
                    count > 1 for count in patient_counts.values()
                ),
                "max_images_per_patient": max(patient_counts.values(), default=0),
                "single_image_assumption_count": sum(
                    member.patient_id_source == "single_image_assumption"
                    for member in included_members
                ),
                "eligible_experiment_a": patient_count >= 10,
                "eligible_experiment_b": patient_count >= 15,
                "experiment_b_max_query_with_7_support_candidates": max(
                    0, min(20, patient_count - 7)
                ),
            }
        )
    return rows


def audit_dataset(
    source_root: Path,
    output_dir: Path,
    class_specs: Sequence[ClassSpec],
    exclusion_rules: dict[str, ExclusionRule] | None = None,
    workers: int = 8,
) -> dict[str, object]:
    """Audit selected source classes and write deterministic CSV/JSON outputs."""

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    exclusion_rules = exclusion_rules or {}
    jobs: list[tuple[Path, ClassSpec]] = []
    for spec in class_specs:
        jobs.extend((path, spec) for path in _iter_class_images(source_root, spec))

    def read_job(job: tuple[Path, ClassSpec]):
        return _read_image_record(job[0], source_root, job[1], exclusion_rules)

    records: list[ImageRecord] = []
    invalid_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for record, invalid in executor.map(read_job, jobs):
            if record is not None:
                records.append(record)
            if invalid is not None:
                invalid_rows.append(invalid)

    records.sort(key=lambda item: (item.class_id, item.source_relpath.casefold()))
    invalid_rows.sort(key=lambda item: (item["class_id"], item["source_relpath"]))
    record_paths = {record.source_relpath for record in records}
    missing_exclusions = sorted(set(exclusion_rules).difference(record_paths))
    if missing_exclusions:
        raise ValueError(
            "Excluded files were not found in the selected classes: "
            + ", ".join(missing_exclusions)
        )
    missing_canonical = sorted(
        {
            rule.canonical_source_relpath
            for rule in exclusion_rules.values()
            if rule.canonical_source_relpath not in record_paths
        }
    )
    if missing_canonical:
        raise ValueError(
            "Canonical duplicate files were not found: " + ", ".join(missing_canonical)
        )
    manifest_rows = [asdict(record) for record in records]
    duplicate_rows = _duplicate_rows(records)
    class_summary_rows = _class_summary_rows(class_specs, records, invalid_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "patient_manifest.csv",
        manifest_rows,
        list(ImageRecord.__dataclass_fields__),
    )
    _write_csv(
        output_dir / "class_patient_summary.csv",
        class_summary_rows,
        list(class_summary_rows[0]) if class_summary_rows else [],
    )
    _write_csv(
        output_dir / "duplicate_candidates.csv",
        duplicate_rows,
        [
            "group_type",
            "group_id",
            "fingerprint",
            "member_count",
            "class_id",
            "class_name",
            "patient_id",
            "source_relpath",
            "include_in_experiments",
            "canonical_source_relpath",
        ],
    )
    _write_csv(
        output_dir / "invalid_images.csv",
        invalid_rows,
        ["class_id", "class_name", "source_relpath", "error"],
    )

    summary = {
        "source_root": str(source_root),
        "selected_class_count": len(class_specs),
        "valid_image_count": len(records),
        "included_image_count": sum(
            record.include_in_experiments for record in records
        ),
        "excluded_image_count": sum(
            not record.include_in_experiments for record in records
        ),
        "invalid_image_count": len(invalid_rows),
        "patient_count_within_classes": sum(
            int(row["patient_count"]) for row in class_summary_rows
        ),
        "single_image_assumption_count": sum(
            record.patient_id_source == "single_image_assumption"
            and record.include_in_experiments
            for record in records
        ),
        "exact_duplicate_group_count": len(
            {row["group_id"] for row in duplicate_rows if row["group_type"] == "exact_sha256"}
        ),
        "equal_dhash_group_count": len(
            {row["group_id"] for row in duplicate_rows if row["group_type"] == "equal_dhash64"}
        ),
        "experiment_a_eligible_class_count": sum(
            bool(row["eligible_experiment_a"]) for row in class_summary_rows
        ),
        "experiment_b_eligible_class_count": sum(
            bool(row["eligible_experiment_b"]) for row in class_summary_rows
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=Path("configs/classes.json"),
    )
    parser.add_argument(
        "--exclusions-config",
        type=Path,
        default=Path("configs/data_exclusions.json"),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = load_class_specs(args.classes_config)
    exclusions = load_exclusion_rules(args.exclusions_config)
    summary = audit_dataset(
        args.source_root,
        args.output_dir,
        specs,
        exclusion_rules=exclusions,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
