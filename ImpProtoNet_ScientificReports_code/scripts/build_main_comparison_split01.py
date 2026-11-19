import csv

import gzip

import hashlib

import json

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

MODELS = {'M0': 'ProtoNet', 'M1': 'FrozenNet', 'M2': 'TransferNet',
          'M3': 'MatchNet', 'M5': 'RelationNet', 'Ours': 'Ours'}

CLASSES = ['K06', 'K08', 'K11', 'K12', 'K20']

def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return {'accuracy': float(tp.sum() / cm.sum()),
            'precision': float(safe(tp, tp + fp).mean()),
            'sensitivity': float(safe(tp, tp + fn).mean()),
            'specificity': float(safe(tn, tn + fp).mean()),
            'macro_f1': float(safe(2 * tp, 2 * tp + fp + fn).mean())}

def verified_cm(rows, episodes, shot):
    expected = {e['episode_id']: e for e in episodes}
    cm = np.zeros((5, 5), dtype=np.int64)
    index = {c: i for i, c in enumerate(CLASSES)}
    seen = set()
    for r in rows:
        assert r['split_id'] == 'split_01' and int(r['shot']) == shot
        key = r['episode_id'], int(r['query_position'])
        assert 1 <= key[1] <= 15
        assert key not in seen
        seen.add(key)
        e = expected[key[0]]
        target, item = [(c['class_id'], x) for c in e['classes'] for x in c['query']][key[1] - 1]
        assert (r['patient_id'], r['image_id']) == (item['patient_id'], item['image_id'])
        assert r.get('target_class_id', r.get('true_class_id')) == target
        assert r['predicted_class_id'] in e['class_ids']
        if 'probabilities_by_class' in r:
            probabilities = r['probabilities_by_class']
            assert set(probabilities) == set(e['class_ids'])
            assert np.isclose(sum(probabilities.values()), 1, atol=1e-5)
            assert np.isclose(probabilities[r['predicted_class_id']], max(probabilities.values()), atol=1e-7)
        cm[index[target], index[r['predicted_class_id']]] += 1
    assert len(rows) == len(seen) == 15000
    assert {key[0] for key in seen} == set(expected)
    assert np.all(cm.sum(1) == 3000)
    return cm

def draw_confusion(result):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
                         'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.4))
    fig.subplots_adjust(left=.07, right=.89, top=.94, bottom=.18, wspace=.26, hspace=.36)
    rows = [r for r in result['results'] if r['shot'] == 5]
    for letter, ax, row in zip('abcdef', axes.flat, rows, strict=True):
        cm = np.array(row['confusion_counts'])
        percentages = 100 * cm / cm.sum(1, keepdims=True)
        im = ax.imshow(percentages, vmin=0, vmax=100, cmap='Blues')
        ax.set_xticks(range(5), CLASSES, fontsize=9)
        ax.set_yticks(range(5), CLASSES, fontsize=9)
        ax.set_xlabel('Predicted class', fontsize=9)
        ax.set_ylabel('True class', fontsize=9)
        ax.set_title(f"({letter}) {row['model']}", fontsize=11, pad=9)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{percentages[i,j]:.1f}', ha='center', va='center',
                        color='white' if percentages[i,j] >= 55 else 'black', fontsize=8)
    color_ax = fig.add_axes([.915, .23, .015, .63])
    fig.colorbar(im, cax=color_ax, label='Percentage within true class (%)')
    fig.text(.5, .085, 'K06: Pterygium; K08: Seborrheic keratosis; K11: Xanthogranuloma\n'
             'K12: Basal cell carcinoma; K20: Papillary epithelial hyperplasia',
             ha='center', va='center', fontsize=10, linespacing=1.5)
    fig.savefig(PROJECT / 'confusion-matrix.pdf')
    preview = PROJECT / 'tmp/pdfs/main_comparison_review'
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / 'confusion-matrix.png', dpi=220)
    plt.close(fig)
