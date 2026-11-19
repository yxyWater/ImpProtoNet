#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)

    training_rows = read_csv(run_dir / "training_history.csv")
    validation_rows = read_csv(run_dir / "validation_history.csv")

    training_episode = [int(row["episode_index"]) for row in training_rows]
    training_loss = [float(row["loss"]) for row in training_rows]
    validation_episode = [int(row["train_episode_index"]) for row in validation_rows]
    validation_loss = [float(row["mean_loss"]) for row in validation_rows]
    validation_accuracy = [float(row["mean_accuracy"]) for row in validation_rows]
    validation_macro_f1 = [float(row["mean_macro_f1"]) for row in validation_rows]

    best = summary["best_validation"]
    best_episode = int(best["train_episode_index"])
    shot = int(summary["shot"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], training_episode, training_loss, "Training loss", "Loss"),
        (axes[0, 1], validation_episode, validation_loss, "Validation loss", "Loss"),
        (axes[1, 0], validation_episode, validation_accuracy, "Validation accuracy", "Accuracy"),
        (axes[1, 1], validation_episode, validation_macro_f1, "Validation Macro-F1", "Macro-F1"),
    ]
    for ax, x, y, title, ylabel in panels:
        ax.plot(x, y, linewidth=1.5)
        ax.axvline(best_episode, color="tab:red", linestyle="--", label=f"Best checkpoint: episode {best_episode}")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"RPAS convergence curves ({shot}-shot, split_01)")
    fig.tight_layout()
    output_path = output_dir / f"convergence_{shot}shot.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    combined_path = output_dir / f"convergence_{shot}shot.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "training_loss",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
        ])
        validation_by_episode = {
            int(row["train_episode_index"]): row for row in validation_rows
        }
        for row in training_rows:
            episode = int(row["episode_index"])
            validation = validation_by_episode.get(episode)
            writer.writerow([
                episode,
                row["loss"],
                validation["mean_loss"] if validation else "",
                validation["mean_accuracy"] if validation else "",
                validation["mean_macro_f1"] if validation else "",
            ])

    print(json.dumps({
        "shot": shot,
        "best_episode": best_episode,
        "figure": str(output_path),
        "csv": str(combined_path),
    }))


if __name__ == "__main__":
    main()
