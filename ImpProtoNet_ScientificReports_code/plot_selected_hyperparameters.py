"""Draw the manuscript's three hyperparameter panels from a user-supplied CSV."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from plot_hyperparameter_effects import PARAMETERS, draw_panel, panel_rows, plt, readcsv


def read_points(path: Path) -> list[dict]:
    points = []
    for row in readcsv(path):
        panel = row["panel"]
        shot = int(row["shot"])
        score = float(row["validation_accuracy_percent"])
        value = float(row["parameter_value"])
        if (panel not in PARAMETERS or shot not in (3, 5) or not 0 <= score <= 100
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid plot row: {row}")
        points.append(dict(panel=panel, shot=shot, parameter_value=value, validation_accuracy_percent=score))
    for panel, (reference, *_) in PARAMETERS.items():
        values = [[row["parameter_value"] for row in panel_rows(points, panel, shot)] for shot in (3, 5)]
        if values[0] != values[1] or len(set(values[0])) != 5 or len(values[0]) != 5 or reference not in values[0]:
            raise ValueError(f"Expected the same five distinct values and a reference for {panel} in both shot settings")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = read_points(args.input)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for letter, ax, panel in zip("abc", axes, PARAMETERS, strict=True):
        draw_panel(ax, points, panel, f"({letter}) ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
