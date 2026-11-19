import csv

import hashlib

import json

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

import numpy as np

MODELS = {'M0': 'ProtoNet', 'M1': 'FrozenNet', 'M2': 'TransferNet',
          'M3': 'MatchNet', 'M5': 'RelationNet', 'Ours': 'Ours'}

def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_panels(SOURCE, predictions):
    bins_path = SOURCE / 'results/reliability_bins.csv'
    metrics_path = SOURCE / 'results/model_metrics.csv'
    bins = readcsv(bins_path)
    metrics = {r['model_id']: r for r in readcsv(metrics_path)}
    panels = []
    for mid, name in MODELS.items():
        rows = sorted((r for r in bins if r['model_id'] == mid and
                       r['scope'] == 'split' and r['split_id'] == 'split_01' and
                       r['analysis_tier'] == 'primary'), key=lambda r: int(r['bin_index']))
        if [int(r['bin_index']) for r in rows] != list(range(1, 11)):
            raise ValueError(f'{mid}: ten bins required')
        counts = np.array([int(r['prediction_count']) for r in rows])
        confidence = np.array([float(r['mean_confidence']) for r in rows])
        accuracy = np.array([float(r['empirical_accuracy']) for r in rows])
        metric = metrics[mid]
        if counts.sum() != 15000 or int(metric['shot']) != 5 or int(metric['split_count']) != 1:
            raise ValueError(f'{mid}: fixed-split 5-shot prediction count mismatch')
        ece = float(np.sum(counts * np.abs(accuracy - confidence)) / counts.sum())
        if not np.isclose(ece, float(metric['ece']), atol=1e-12, rtol=0):
            raise ValueError(f'{mid}: ECE differs from Table 6')
        if not np.isclose(np.dot(counts, accuracy) / counts.sum(), float(metric['accuracy']),
                          atol=1e-12, rtol=0):
            raise ValueError(f'{mid}: accuracy differs from Table 6')
        path = predictions / mid / 'split_01/5shot/test_query_predictions.jsonl.gz'
        panels.append(dict(model_id=mid, model=name, counts=counts.tolist(),
                           mean_confidence=confidence.tolist(), accuracy=accuracy.tolist(),
                           ece=ece, prediction_count=int(counts.sum()),
                           prediction_source=str(path.relative_to(SOURCE)),
                           prediction_sha256=sha(path)))
    return panels, {str(p.relative_to(SOURCE)): sha(p) for p in (bins_path, metrics_path)}

def curve_values(panel):
    # Keep empty bins as NaNs so lines never bridge unobserved confidence ranges.
    present = np.asarray(panel['counts']) > 0
    return (np.where(present, np.asarray(panel['mean_confidence']) * 100, np.nan),
            np.where(present, np.asarray(panel['accuracy']) * 100, np.nan))

def render(panels, path):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.4), layout='constrained')
    for index, (ax, panel) in enumerate(zip(axes.flat, panels)):
        ax.bar(np.arange(5, 100, 10), np.asarray(panel['counts']) / 150,
               width=8.5, color='#BDDDF2', edgecolor='#7EB4D7', linewidth=.5, zorder=2)
        ax.plot([0, 100], [0, 100], '--', color='#737373', linewidth=1, zorder=1)
        x, y = curve_values(panel)
        ax.plot(x, y, '-o', color='#C43D37', linewidth=1.4, markersize=4, zorder=3)
        ax.set(xlim=(0, 101), ylim=(0, 104), xticks=np.arange(0, 101, 20),
               yticks=np.arange(0, 101, 20), xlabel='Confidence (%)')
        ax.set_title(f"({chr(97 + index)}) {panel['model']}\nECE = {panel['ece'] * 100:.1f}%",
                     fontsize=12, pad=8)
        if index % 3 == 0:
            ax.set_ylabel('Accuracy / predictions (%)')
        ax.grid(axis='y', color='#E2E2E2', linewidth=.6, zorder=0)
        ax.tick_params(labelsize=10)
    handles = [Patch(facecolor='#BDDDF2', edgecolor='#7EB4D7', label='Predictions per bin'),
               Line2D([], [], color='#C43D37', marker='o', markersize=4, label='Observed accuracy'),
               Line2D([], [], color='#737373', linestyle='--', label='Perfect calibration')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False,
               fontsize=11)
    fig.savefig(path, metadata={'Title': 'Reliability diagrams under the 5-shot setting',
                               'Subject': 'Six models; fixed disease split; Table 6 prediction sources'})
    plt.close(fig)
