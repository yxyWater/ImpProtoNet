import hashlib

import json

from collections import Counter, defaultdict

from pathlib import Path



CLASSES = ['K06', 'K08', 'K11', 'K12', 'K20']

METRICS = ['accuracy', 'precision', 'specificity', 'macro_f1']

def metrics(rows):
    """Pool queries within one repeat, then macro-average across five diseases."""
    if not rows:
        raise ValueError('Cannot summarize empty predictions')
    cm = Counter((r['true_class_id'], r['predicted_class_id']) for r in rows)
    if any(a not in CLASSES or b not in CLASSES for a, b in cm):
        raise ValueError('Unexpected disease class')
    detail = []
    for c in CLASSES:
        tp = cm[c, c]
        fp = sum(n for (a, b), n in cm.items() if b == c and a != c)
        fn = sum(n for (a, b), n in cm.items() if a == c and b != c)
        tn = len(rows) - tp - fp - fn
        safe = lambda a, b: a / b if b else 0.0
        detail.append([safe(tp, tp + fp), safe(tn, tn + fp), safe(2 * tp, 2 * tp + fp + fn)])
    return dict(accuracy=sum(cm[c, c] for c in CLASSES) / len(rows),
                **{k: sum(d[j] for d in detail) / 5
                   for j, k in enumerate(METRICS[1:])})

def summarize(rows, episodes, shot):
    expected = {e['episode_id']: e for e in episodes}
    groups = defaultdict(list)
    seen = set()
    query_panels = {}
    for row in rows:
        e = expected[row['episode_id']]
        repeat, combo = e['support_resample_index'], e['disease_combination_index']
        position = int(row['query_position'])
        assert 1 <= position <= 15
        key = repeat, combo, position
        assert key not in seen
        seen.add(key)
        assert row['split_id'] == 'split_01' and int(row['shot']) == shot
        assert int(row['repeat']) == repeat and int(row['combination']) == combo
        truth, item = [(c['class_id'], x) for c in e['classes'] for x in c['query']][position - 1]
        identity = truth, item['patient_id'], item['image_id']
        assert (row['true_class_id'], row['patient_id'], row['image_id']) == identity
        assert row['predicted_class_id'] in e['class_ids']
        assert query_panels.setdefault((combo, position), identity) == identity
        groups[repeat].append(row)
    assert len(seen) == 15150 and set(groups) == set(range(101))
    assert all(len(v) == 150 for v in groups.values())
    repeat_metrics = []
    for repeat in range(101):
        assert Counter(r['true_class_id'] for r in groups[repeat]) == Counter({c: 30 for c in CLASSES})
        repeat_metrics.append(dict(repeat=repeat, **metrics(groups[repeat])))
    baseline = {k: repeat_metrics[0][k] for k in METRICS}
    resampled = {k: sum(r[k] for r in repeat_metrics[1:]) / 100 for k in METRICS}
    delta = {k: resampled[k] - baseline[k] for k in METRICS}
    return baseline, resampled, delta, repeat_metrics
