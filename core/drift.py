import json
import os
from typing import Optional

STATS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_stats.json')


def save_training_stats(df, feature_cols: list):
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std':  float(df[col].std()),
                'min':  float(df[col].min()),
                'max':  float(df[col].max()),
            }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f)


def load_training_stats() -> dict:
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH) as f:
        return json.load(f)


def detect_drift(current_inputs: dict, threshold: float = 2.0) -> dict:
    stats = load_training_stats()
    report = {}
    for feature, value in current_inputs.items():
        if feature not in stats or stats[feature]['std'] == 0:
            continue
        s       = stats[feature]
        z_score = abs((value - s['mean']) / s['std'])
        report[feature] = {
            'value':      value,
            'train_mean': round(s['mean'], 3),
            'train_std':  round(s['std'], 3),
            'z_score':    round(z_score, 2),
            'drifted':    z_score > threshold,
        }
    return report
