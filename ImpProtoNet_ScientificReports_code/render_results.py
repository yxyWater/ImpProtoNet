"""Redraw numerical manuscript figures from the included aggregate CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_main_comparison_split01 as comparison
import plot_reliability_diagrams_split01 as reliability


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def confusion(data: Path, output: Path) -> None:
    rows = read_csv(data / "figure2_confusion_counts.csv")
    result = []
    for model, label in comparison.MODELS.items():
        selected = {row["true_class_id"]: row for row in rows if row["model_id"] == model}
        matrix = [[int(selected[truth][prediction]) for prediction in comparison.CLASSES]
                  for truth in comparison.CLASSES]
        result.append(dict(model="ImpProtoNet" if model == "Ours" else label,
                           shot=5, confusion_counts=matrix))
    comparison.PROJECT = output
    comparison.draw_confusion({"results": result})


def confidence_figures(data: Path, output: Path) -> None:
    metrics = {row["model_id"]: row for row in read_csv(data / "table5_reliability.csv")}
    bins = read_csv(data / "supp_figure3_reliability_bins.csv")
    curves = read_csv(data / "figure3_risk_coverage.csv")
    panels = []
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for model, label in comparison.MODELS.items():
        label = "ImpProtoNet" if model == "Ours" else label
        selected = sorted(
            (row for row in bins if row["model_id"] == model and row["scope"] == "split"),
            key=lambda row: int(row["bin_index"]),
        )
        panels.append(dict(model=label, counts=[int(row["prediction_count"]) for row in selected],
                           mean_confidence=[float(row["mean_confidence"]) for row in selected],
                           accuracy=[float(row["empirical_accuracy"]) for row in selected],
                           ece=float(metrics[model]["ece"])))
        selected = sorted(
            (row for row in curves if row["model_id"] == model and row["scope"] == "pooled"),
            key=lambda row: float(row["actual_coverage"]),
        )
        ax.plot([100 * float(row["actual_coverage"]) for row in selected],
                [100 * float(row["risk"]) for row in selected],
                label=f"{label} (AURC = {float(metrics[model]['aurc']):.3f})")
    ax.set(xlabel="Coverage (%)", ylabel="Risk (error rate, %)")
    ax.legend(frameon=False)
    fig.savefig(output / "risk-coverage.pdf")
    plt.close(fig)
    reliability.render(panels, output / "reliability_diagrams.pdf")


def convergence(data: Path, output: Path, shot: int) -> None:
    rows = read_csv(data / f"supp_figure1_convergence_{shot}shot.csv")
    fields = (
        ("training_loss", "Training loss"), ("validation_loss", "Validation loss"),
        ("validation_accuracy", "Validation accuracy"), ("validation_macro_f1", "Validation macro-F1"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), layout="constrained")
    for ax, (field, title) in zip(axes.flat, fields, strict=True):
        measured = [row for row in rows if row[field] != ""]
        ax.plot([int(row["episode"]) for row in measured], [float(row[field]) for row in measured])
        ax.set(title=title, xlabel="Training episode")
    fig.savefig(output / f"convergence_{shot}shot.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    confusion(args.data, args.output)
    confidence_figures(args.data, args.output)
    for shot in (3, 5):
        convergence(args.data, args.output, shot)
    subprocess.run(
        [sys.executable, str(ROOT / "plot_selected_hyperparameters.py"),
         "--input", str(args.data / "supp_figure2_hyperparameters.csv"),
         "--output", str(args.output / "super-parameters.pdf")], check=True,
    )


if __name__ == "__main__":
    main()
