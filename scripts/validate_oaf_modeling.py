#!/usr/bin/env python3
"""Validate OAF modeling outputs; optionally compare independently generated standalone CSV.

Usage: python scripts/validate_oaf_modeling.py --run-id oaf-modeling-001
       python scripts/validate_oaf_modeling.py --run-id oaf-modeling-001 --standalone-predictions PATH

Standalone CSV requires dtxsid, model, y_true, y_pred columns. It must be produced
by a separate standalone pipeline run, not copied from the OAF output.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REQUIRED = {'dtxsid', 'model', 'y_true', 'y_pred'}


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(REQUIRED - set(reader.fieldnames or []))}')
        rows = {}
        for r in reader:
            key = (r['model'], r['dtxsid'])
            if key in rows:
                raise ValueError(f'{path}: duplicate model/compound {key}')
            rows[key] = (float(r['y_true']), float(r['y_pred']))
            if not all(math.isfinite(x) for x in rows[key]):
                raise ValueError(f'{path}: nonfinite prediction {key}')
    return rows


def paired_bootstrap(rows, left='D', right='A', iterations=10000, seed=42):
    ids_left = {sid for model, sid in rows if model == left}
    ids_right = {sid for model, sid in rows if model == right}
    if ids_left != ids_right or not ids_left:
        raise ValueError('Paired models must contain identical, nonempty compound IDs')
    ids = sorted(ids_left)
    y = np.array([rows[(left, sid)][0] for sid in ids])
    y_right = np.array([rows[(right, sid)][0] for sid in ids])
    if not np.allclose(y, y_right, atol=1e-8, rtol=0):
        raise ValueError('Models have inconsistent observed targets')
    a = np.array([rows[(left, sid)][1] for sid in ids])
    b = np.array([rows[(right, sid)][1] for sid in ids])
    rng = np.random.default_rng(seed)
    diffs_mae, diffs_rmse, diffs_rho = [], [], []
    for _ in range(iterations):
        ix = rng.integers(0, len(ids), size=len(ids))
        yy, aa, bb = y[ix], a[ix], b[ix]
        diffs_mae.append(float(np.mean(abs(yy-aa)) - np.mean(abs(yy-bb))))
        diffs_rmse.append(float(np.sqrt(np.mean((yy-aa)**2)) - np.sqrt(np.mean((yy-bb)**2))))
        if len(set(yy)) > 1 and len(set(aa)) > 1 and len(set(bb)) > 1:
            diffs_rho.append(float(spearmanr(yy, aa).statistic - spearmanr(yy, bb).statistic))
    def summary(vals):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None, 'valid_resamples': 0}
        return {'mean': float(np.mean(vals)), 'ci_lo': float(np.percentile(vals, 2.5)),
                'ci_hi': float(np.percentile(vals, 97.5)), 'valid_resamples': len(vals)}
    return {'comparison': f'{left} minus {right}', 'n': len(ids),
            'delta_mae': summary(diffs_mae), 'delta_rmse': summary(diffs_rmse),
            'delta_spearman_rho': summary(diffs_rho),
            'interpretation': 'Negative MAE/RMSE deltas favor first model; positive Spearman delta favors first model.'}


def compare_predictions(oaf, standalone, atol=1e-6):
    if set(oaf) != set(standalone):
        missing = sorted(set(oaf) - set(standalone))[:10]
        extra = sorted(set(standalone) - set(oaf))[:10]
        return {'pass': False, 'reason': 'model/compound keys differ', 'missing_in_standalone': missing,
                'extra_in_standalone': extra}
    max_y = max(abs(oaf[k][0]-standalone[k][0]) for k in oaf)
    max_pred = max(abs(oaf[k][1]-standalone[k][1]) for k in oaf)
    return {'pass': max_y <= atol and max_pred <= atol, 'n_predictions': len(oaf),
            'max_abs_target_difference': max_y, 'max_abs_prediction_difference': max_pred,
            'absolute_tolerance': atol}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--runs-root', default='runs/oaf-work')
    p.add_argument('--standalone-predictions', type=Path)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--atol', type=float, default=1e-6)
    args = p.parse_args()
    if args.iterations < 1 or args.atol < 0:
        p.error('iterations must be positive and atol nonnegative')
    run = Path(args.runs_root) / args.run_id
    report = json.loads((run/'metrics.json').read_text())
    rows = load_rows(run/'predictions.csv')
    models = sorted({model for model, _ in rows})
    n_by_model = {model: sum(k[0] == model for k in rows) for model in models}
    if set(models) != {'A','B','D'} or len(set(n_by_model.values())) != 1:
        raise ValueError(f'Unexpected model coverage: {n_by_model}')
    ids = [{sid for m, sid in rows if m == model} for model in models]
    if not all(s == ids[0] for s in ids):
        raise ValueError('Model test compound sets differ')
    if not isinstance(report, dict) or not {'A','B','D'}.issubset(report):
        raise ValueError('metrics.json missing A/B/D metrics')
    outcome = {'run_id': args.run_id, 'models': n_by_model,
               'paired_bootstrap': paired_bootstrap(rows, iterations=args.iterations, seed=args.seed),
               'standalone_equivalence': {'status': 'NOT_TESTED', 'reason': 'No independently generated standalone predictions supplied'}}
    if args.standalone_predictions:
        result = compare_predictions(rows, load_rows(args.standalone_predictions), args.atol)
        outcome['standalone_equivalence'] = {'status': 'PASS' if result['pass'] else 'FAIL', **result}
    dest = run/'validation.json'
    dest.write_text(json.dumps(outcome, indent=2, sort_keys=True)+'\n')
    print(json.dumps(outcome, indent=2))
    if outcome['standalone_equivalence']['status'] == 'FAIL':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
