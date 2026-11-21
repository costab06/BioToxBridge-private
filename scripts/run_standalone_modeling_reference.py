#!/usr/bin/env python3
"""Recompute A/B/D predictions directly from a frozen OAF dataset split.

Independent execution path, shared scientific projector and trainer implementations.
Does not invoke OAF model_training, model_evaluation or reporting stages.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml


def find_split(artifact_root: Path, run_id: str) -> Path:
    folder = artifact_root / run_id
    candidates = [p for p in folder.glob('dataset_split-*.json')
                  if not p.name.endswith('-input-binding.json')]
    if len(candidates) != 1:
        raise ValueError(f'Expected one immutable dataset_split output in {folder}; found {len(candidates)}')
    return candidates[0]


def reference_rows(split: dict, config: dict) -> list[dict]:
    from biotoxbridge.datasets.projectors import project_A, project_B, project_D
    from biotoxbridge.training.trainer import train_model
    train, test = split['train'], split['test']
    if not train or not test:
        raise ValueError('Frozen train and test partitions must be nonempty')
    fp = config['chemistry']['fingerprint']
    projectors = {
        'A': lambda: project_A(train, test, radius=fp['radius'], n_bits=fp['bits']),
        'B': lambda: project_B(train, test),
        'D': lambda: project_D(train, test, radius=fp['radius'], n_bits=fp['bits']),
    }
    model_map = {'structure_only': 'A', 'predicted_bioactivity': 'B',
                 'structure_plus_predicted': 'D'}
    requested = [model_map.get(m, m) for m in config['models']]
    if set(requested) != {'A', 'B', 'D'} or len(requested) != 3:
        raise ValueError('Reference requires models A, B, D exactly once')
    rows = []
    for mid in requested:
        X_train, X_test, y_train, y_test, ids_train, ids_test, meta = projectors[mid]()
        if len(ids_train) != len(train) or len(ids_test) != len(test):
            raise ValueError(f'Model {mid}: projector changed partition size')
        model, info = train_model(X_train, y_train, model_id=mid)
        scaler = meta['y_scaler']
        actual = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1)).ravel()
        predicted = scaler.inverse_transform(np.asarray(model.predict(X_test)).reshape(-1, 1)).ravel()
        print(f'Model {mid}: alpha={info["best_alpha"]:.8g}; test={len(ids_test)}')
        rows.extend({'dtxsid': str(sid), 'model': mid, 'y_true': float(y), 'y_pred': float(p)}
                    for sid, y, p in zip(ids_test, actual, predicted))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', type=Path, default=Path('config/experiments/liver_loael_v1.yaml'))
    parser.add_argument('--artifact-root', type=Path, default=Path('runs/oaf-artifacts'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    split_file = find_split(args.artifact_root, args.run_id)
    split = json.loads(split_file.read_text())
    config = yaml.safe_load(args.config.read_text())
    output = args.output or Path('runs/standalone-validation') / args.run_id / 'predictions.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = reference_rows(split, config)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('dtxsid', 'model', 'y_true', 'y_pred'))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} standalone predictions to {output}')
    print('NOTE: This compares execution paths using shared projectors/trainer and the same frozen split; it is not an independent dataset preparation validation.')


if __name__ == '__main__':
    main()
