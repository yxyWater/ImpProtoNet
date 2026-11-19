import csv

import hashlib

import json

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PARAMETERS = {
    "learning_rate": (0.0001, "Learning rate", "Learning rate", "learning-rate"),
    "weight_decay": (0.001, "Weight decay", "Weight decay", "weight-decay"),
    "focal_gamma": (1.5, "Focal loss focusing parameter", r"Focusing parameter $\gamma$", "focal-gamma"),
}

def readcsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def panel_rows(points, parameter, shot):
    return sorted((p for p in points if p["panel"] == parameter and p["shot"] == shot),
                  key=lambda p: p["parameter_value"])

def tick_label(value, parameter):
    if parameter == "learning_rate" or (parameter == "weight_decay" and value < 0.01):
        coefficient, exponent = f"{value:.0e}".split("e")
        prefix = "" if coefficient == "1" else coefficient + r"\times"
        return f"${prefix}10^{{{int(exponent)}}}$"
    return f"{value:g}"

def draw_panel(ax, points, parameter, letter=""):
    _, title, xlabel, _ = PARAMETERS[parameter]
    values = [p["parameter_value"] for p in panel_rows(points, parameter, 3)]
    positions = {value: index for index, value in enumerate(values)}
    for shot, color, marker in [(3, "#2166ac", "o"), (5, "#d95f02", "s")]:
        rows = panel_rows(points, parameter, shot)
        ax.plot([positions[p["parameter_value"]] for p in rows],
                [p["validation_accuracy_percent"] for p in rows],
                marker=marker, color=color, markerfacecolor=color, linestyle="-",
                linewidth=1.7, markersize=5.5, label=f"{shot}-shot")
    ax.set_xticks(range(len(values)), [tick_label(value, parameter) for value in values])
    ax.set_ylim(55, 84)
    ax.set_yticks(range(55, 81, 5))
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_title(f"{letter}{title}", fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.margins(x=0.10)
