"""Manifest-level patient/image exposure balancing for training episodes."""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

def _load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                return [json.loads(line) for line in text if line.strip()]

def _save_manifest(path: Path, episodes: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for episode in episodes:
                    text.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
                    text.write("\n")

def _split_manifest_for(source: Path) -> Path:
    return source.resolve().parents[2] / "class_splits" / "split_image_manifest.csv"

def load_patient_pool(source: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    rows = list(csv.DictReader(_split_manifest_for(source).open(encoding="utf-8-sig", newline="")))
    split_id = source.parent.name
    pool: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split_id"] != split_id or row["partition"] != "train":
            continue
        pool[row["class_id"]][row["patient_id"]].append(
            {
                "patient_id": row["patient_id"],
                "image_id": row["image_id"],
                "source_relpath": row["source_relpath"],
            }
        )
    for patients in pool.values():
        for images in patients.values():
            images.sort(key=lambda item: item["source_relpath"].casefold())
    return pool

def _select_least_used(candidates: list[str], exposure: Counter[str], rng: random.Random) -> str:
    minimum = min(exposure[candidate] for candidate in candidates)
    tied = [candidate for candidate in candidates if exposure[candidate] == minimum]
    return rng.choice(tied)

def _choose_image(images: list[dict[str, str]], image_exposure: Counter[str], rng: random.Random) -> dict[str, str]:
    minimum = min(image_exposure[item["source_relpath"]] for item in images)
    tied = [item for item in images if image_exposure[item["source_relpath"]] == minimum]
    return dict(rng.choice(tied))


def balanced_image_replacement_manifest(source: Path, destination: Path, *, preserve_query: bool = False) -> dict[str, object]:
    """Balance image exposure while preserving episode patient structure."""
    import csv
    

    split_id=source.parent.name
    pool: dict[str, dict[str, list[str]]] = {}
    with _split_manifest_for(source).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split_id"] != split_id or row["partition"] != "train":
                continue
            pool.setdefault(row["class_id"], {}).setdefault(row["patient_id"], []).append(row["source_relpath"])
    for cls in pool:
        for patient in pool[cls]:
            pool[cls][patient].sort(key=lambda x: x.casefold())
    episodes=_load_manifest(source)
    rng=random.Random(20261004)
    image_exposure: Counter[str]=Counter()
    missing=0
    for episode in episodes:
        for class_item in episode["classes"]:
            cls=str(class_item["class_id"])
            for item in class_item["support"] + ([] if preserve_query else class_item["query"]):
                patient=str(item["patient_id"])
                candidates=pool.get(cls, {}).get(patient)
                if not candidates:
                    missing += 1
                    continue
                chosen=_select_least_used(candidates, image_exposure, rng)
                item["source_relpath"]=chosen
                image_exposure[chosen] += 1
    _save_manifest(destination,episodes)
    return {"source":str(source), "destination":str(destination), "episode_count":len(episodes), "split_id":split_id, "selection_rule":"fixed_original_patients_least_used_image" + ("_support_only" if preserve_query else "_support_and_query"), "missing_class_patient_records":missing, "unique_images_used":len(image_exposure)}

def balance_manifest_by_image(
    source: Path,
    destination: Path,
    *,
    mix_probability: float = 1.0,
) -> dict[str, object]:
    """Create a deterministic patient-balanced training manifest.

    A probability of 1.0 replaces every support/query patient in the source
    episode with a least-used eligible patient. Values below 1.0 keep that
    item's original patient unchanged and form a mixed standard/balanced
    schedule. Image exposure remains balanced for replaced patients.
    """

    if destination.resolve() == source.resolve():
        raise ValueError("Destination must differ from source")
    if not 0.0 <= mix_probability <= 1.0:
        raise ValueError("mix_probability must be in [0, 1]")
    pool = load_patient_pool(source)
    episodes = _load_manifest(source)
    rng = random.Random(20261004)
    patient_exposure: Counter[str] = Counter()
    image_exposure: Counter[str] = Counter()
    replaced_items = 0
    kept_items = 0

    for episode in episodes:
        for class_item in episode["classes"]:
            class_id = str(class_item["class_id"])
            patients = pool[class_id]
            used_patient_ids = {
                str(item["patient_id"]) for role in ("support", "query") for item in class_item[role]
            }
            if len(used_patient_ids) != sum(len(class_item[role]) for role in ("support", "query")):
                raise RuntimeError(
                    f"Episode {episode['episode_id']} class {class_id} reuses a patient within an episode"
                )
            for role in ("support", "query"):
                for rank, item in enumerate(class_item[role], start=1):
                    if mix_probability < 1.0 and rng.random() >= mix_probability:
                        patient_id = str(item["patient_id"])
                        patient_exposure[patient_id] += 1
                        image_exposure[str(item["source_relpath"])] += 1
                        kept_items += 1
                        continue
                    available_patient_ids = sorted(set(patients) - used_patient_ids)
                    if not available_patient_ids:
                        raise RuntimeError(
                            f"Class {class_id} has too few patients outside episode {episode['episode_id']}"
                        )
                    patient_id = _select_least_used(available_patient_ids, patient_exposure, rng)
                    chosen = _choose_image(patients[patient_id], image_exposure, rng)
                    item.update(chosen)
                    item[f"{role}_rank"] = rank
                    if role == "support":
                        item["used_in_shots"] = [
                            shot for shot in (3, 5) if rank <= shot
                        ]
                    used_patient_ids.add(patient_id)
                    patient_exposure[patient_id] += 1
                    image_exposure[chosen["source_relpath"]] += 1
                    replaced_items += 1

    _save_manifest(destination, episodes)
    return {
        "source": str(source),
        "destination": str(destination),
        "episode_count": len(episodes),
        "split_id": source.parent.name,
        "selection_rule": "itemwise_least_used_patient_and_image_mixed",
        "mix_probability": mix_probability,
        "replaced_items": replaced_items,
        "kept_original_items": kept_items,
    }

def balance_manifest_by_patient(
    source: Path,
    destination: Path,
    *,
    mix_probability: float = 1.0,
    preserve_query: bool = False,
) -> dict[str, object]:
    """Create a patient-balanced training manifest.

    Patients are chosen with a least-used exposure rule. When
    ``preserve_query`` is true, only support patients are rebalanced and
    query patients/images remain as in the source manifest. For a mixed
    schedule, each item is independently kept from the source with
    probability ``1 - mix_probability`` and otherwise replaced.
    """

    if destination.resolve() == source.resolve():
        raise ValueError("Destination must differ from source")
    if not 0.0 <= mix_probability <= 1.0:
        raise ValueError("mix_probability must be in [0, 1]")
    pool = load_patient_pool(source)
    episodes = _load_manifest(source)
    rng = random.Random(20261004)
    patient_exposure: Counter[str] = Counter()
    replaced_items = 0
    kept_items = 0

    roles = ("support",) if preserve_query else ("support", "query")
    for episode in episodes:
        for class_item in episode["classes"]:
            class_id = str(class_item["class_id"])
            patients = pool[class_id]
            used_patient_ids = {
                str(item["patient_id"])
                for role in ("support", "query")
                for item in class_item[role]
            }
            if len(used_patient_ids) != sum(
                len(class_item[role]) for role in ("support", "query")
            ):
                raise RuntimeError(
                    f"Episode {episode['episode_id']} class {class_id} reuses a patient within an episode"
                )
            for role in roles:
                for rank, item in enumerate(class_item[role], start=1):
                    if mix_probability < 1.0 and rng.random() >= mix_probability:
                        patient_exposure[str(item["patient_id"])] += 1
                        kept_items += 1
                        continue
                    available_patient_ids = sorted(set(patients) - used_patient_ids)
                    if not available_patient_ids:
                        raise RuntimeError(
                            f"Class {class_id} has too few patients outside episode {episode['episode_id']}"
                        )
                    patient_id = _select_least_used(
                        available_patient_ids, patient_exposure, rng
                    )
                    # Keep the original image when a patient is retained by the
                    # mix rule; otherwise use the patient's first image in a
                    # stable order. This balances patient exposure without
                    # changing the image-identity objective.
                    chosen = dict(patients[patient_id][0])
                    item.update(chosen)
                    item[f"{role}_rank"] = rank
                    if role == "support":
                        item["used_in_shots"] = [
                            shot for shot in (3, 5) if rank <= shot
                        ]
                    used_patient_ids.add(patient_id)
                    patient_exposure[patient_id] += 1
                    replaced_items += 1

    _save_manifest(destination, episodes)
    return {
        "source": str(source),
        "destination": str(destination),
        "episode_count": len(episodes),
        "split_id": source.parent.name,
        "selection_rule": "itemwise_least_used_patient"
        + ("_support_only" if preserve_query else "_support_and_query"),
        "mix_probability": mix_probability,
        "replaced_items": replaced_items,
        "kept_original_items": kept_items,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mix-probability", type=float, default=1.0)
    args = parser.parse_args()
    print(
        json.dumps(
            balance_manifest_by_image(
                args.source,
                args.destination,
                mix_probability=args.mix_probability,
            ),
            indent=2,
        )
    )
