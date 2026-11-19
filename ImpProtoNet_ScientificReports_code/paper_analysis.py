"""Recalculate reported metrics from predictions kept with the restricted inputs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from build_main_comparison_split01 import MODELS, metrics_from_cm, verified_cm
from summarize_six_support_robustness import ESTIMANDS, bootstrap, loadgz, summarize_run
from summarize_support_robustness_changes import summarize


def predictions(folder: Path, model: str, shot: int) -> Path:
    return folder / model / "split_01" / f"{shot}shot" / "test_query_predictions.jsonl.gz"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standard_rows(path: Path, episodes: list[dict], *, require_probabilities: bool = True) -> list[dict]:
    classes = {episode["episode_id"]: episode["class_ids"] for episode in episodes}
    rows = loadgz(path)
    for row in rows:
        if "query_position" not in row:
            row["query_position"] = int(row["query_index"]) + 1
        if "probabilities_by_class" not in row:
            if not require_probabilities and "probabilities" not in row:
                continue
            order = row.get("class_ids", classes[row["episode_id"]])
            if order != classes[row["episode_id"]]:
                raise ValueError("Probability order disagrees with the episode class order")
            row["probabilities_by_class"] = dict(zip(order, row["probabilities"], strict=True))
    return rows


def recognition(args: argparse.Namespace) -> None:
    episodes = loadgz(args.manifest)
    rows = []
    model_ids = MODELS if not args.ablation else {
        model: model for model in ("A", "P", "S", "R", "Ours")
    }
    for model, label in model_ids.items():
        for shot in (3, 5):
            records = standard_rows(predictions(args.predictions, model, shot), episodes,
                                    require_probabilities=not args.ablation)
            matrix = verified_cm(records, episodes, shot)
            rows.append(dict(model_id=model, model=label, shot=shot,
                             **metrics_from_cm(matrix), confusion_counts=matrix.tolist()))
    write_csv(args.output / "recognition_metrics.csv",
              [{k: v for k, v in row.items() if k != "confusion_counts"} for row in rows])
    (args.output / "confusion_counts.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    if not args.ablation:
        import build_main_comparison_split01 as plot
        plot.PROJECT = args.output
        rows[-1]["model"] = "ImpProtoNet"
        plot.draw_confusion({"results": rows})
    else:
        keys = ("accuracy", "precision", "sensitivity", "specificity", "macro_f1")
        originals = {row["shot"]: row for row in rows if row["model_id"] == "Ours"}
        write_csv(args.output / "ablation_drops.csv", [
            dict(model_id=row["model_id"], shot=row["shot"],
                 **{f"drop_{key}": originals[row["shot"]][key] - row[key] for key in keys})
            for row in rows
        ])


def support(args: argparse.Namespace) -> None:
    episodes = loadgz(args.manifest)
    manifest = {episode["episode_id"]: episode for episode in episodes}
    draws = np.random.default_rng(20261004).integers(0, 10, size=(10000, 10))
    summary, changes = [], []
    for model, name in MODELS.items():
        for shot in (3, 5):
            path = predictions(args.predictions, model, shot)
            rows = loadgz(path)
            baseline_path = path.with_name("matched_baseline_predictions.jsonl.gz")
            if baseline_path.is_file():
                rows.extend(loadgz(baseline_path))
            for row in rows:
                episode = manifest[row["episode_id"]]
                row.setdefault("repeat", episode["support_resample_index"])
                row.setdefault("combination", episode["disease_combination_index"])
                if "query_position" not in row:
                    row["query_position"] = int(row["query_index"]) + 1
                if "true_class_id" not in row:
                    row["true_class_id"] = row["target_class_id"]
            grid, _, _ = summarize_run(rows, episodes, shot)
            means, bounds, _ = bootstrap(grid, draws)
            for i, name_of_metric in enumerate(ESTIMANDS):
                summary.append(dict(model=name, shot=shot, metric=name_of_metric,
                                    value=means[i], ci_lower=bounds[0, i], ci_upper=bounds[1, i]))
            baseline, resampled, delta, _ = summarize(rows, episodes, shot)
            for metric in delta:
                changes.append(dict(model=name, shot=shot, metric=metric, baseline=baseline[metric],
                                    resampled=resampled[metric], difference=delta[metric]))
    write_csv(args.output / "support_stability.csv", summary)
    write_csv(args.output / "support_changes.csv", changes)


def reliability(args: argparse.Namespace) -> None:
    from eyelid_fsl.calibration_analysis import analyse_calibration, load_calibration_protocol
    import plot_reliability_diagrams_split01 as diagram

    protocol = replace(load_calibration_protocol(ROOT / "configs/calibration_analysis.json"),
                       model_ids=tuple(MODELS), primary_model_ids=tuple(MODELS), supplementary_model_ids=())
    out = args.output / "results"
    normalized = args.output / "normalized"
    episodes = loadgz(args.manifest)
    for model in MODELS:
        rows = standard_rows(predictions(args.predictions, model, 5), episodes)
        verified_cm(rows, episodes, 5)
        dst = predictions(normalized, model, 5)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dst, "wt", encoding="utf-8") as stream:
            for row in rows:
                probability = row["probabilities_by_class"]
                truth = row.get("target_class_id", row.get("true_class_id"))
                row.update(model_id=model, partition="test", target_class_id=truth,
                           confidence=max(probability.values()), true_class_probability=probability[truth],
                           correct=row["predicted_class_id"] == truth)
                stream.write(json.dumps(row) + "\n")
    analyse_calibration(results_root=normalized, output_dir=out, protocol=protocol, split_ids=("split_01",))
    panels, _ = diagram.load_panels(args.output, normalized)
    panels[-1]["model"] = "ImpProtoNet"
    diagram.render(panels, args.output / "reliability_diagrams.pdf")
    metrics = {row["model_id"]: row for row in diagram.readcsv(out / "model_metrics.csv")}
    curves = diagram.readcsv(out / "risk_coverage.csv")
    fig, ax = diagram.plt.subplots(figsize=(7, 4), layout="constrained")
    for model, name in MODELS.items():
        rows = sorted((row for row in curves if row["model_id"] == model and row["scope"] == "pooled"),
                      key=lambda row: float(row["actual_coverage"]))
        if model == "Ours":
            name = "ImpProtoNet"
        ax.plot([100 * float(row["actual_coverage"]) for row in rows],
                [100 * float(row["risk"]) for row in rows],
                label=f"{name} (AURC = {float(metrics[model]['aurc']):.3f})")
    ax.set(xlabel="Coverage (%)", ylabel="Risk (error rate, %)")
    ax.legend(frameon=False)
    fig.savefig(args.output / "risk-coverage.pdf")
    diagram.plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, handler in (("recognition", recognition), ("support", support), ("reliability", reliability)):
        p = sub.add_parser(name)
        p.add_argument("--predictions", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--manifest", required=True, type=Path)
        if name == "recognition":
            p.add_argument("--ablation", action="store_true")
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.handler(args)


if __name__ == "__main__":
    main()
