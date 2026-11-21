#!/usr/bin/env python3
"""Standalone CLI; OAF orchestration will wrap these pure lifecycle operations."""
import argparse
import json
from biotoxbridge.solubility.lifecycle import (ESOL_URL, ModelRegistry, acquire_csv,
    load_snapshot, create_frozen_holdout, train_and_evaluate)

p = argparse.ArgumentParser()
p.add_argument('--store', default='runs/esol-lifecycle')
sub = p.add_subparsers(dest='command', required=True)
a = sub.add_parser('acquire')
a.add_argument('--source', default=ESOL_URL)
a.add_argument('--sha256')
t = sub.add_parser('train')
t.add_argument('--dataset-sha256', required=True)
t.add_argument('--seed', type=int, default=42)
t.add_argument('--promote', action='store_true')
t.add_argument('--max-mae', type=float, default=1.0)
t.add_argument('--min-r2', type=float, default=0.2)
t.add_argument('--max-mae-regression', type=float, default=0.0)
pr = sub.add_parser('promote')
pr.add_argument('version')
pr.add_argument('--max-mae', type=float, default=1.0)
pr.add_argument('--min-r2', type=float, default=0.2)
pr.add_argument('--max-mae-regression', type=float, default=0.0)
i = sub.add_parser('infer')
i.add_argument('smiles')
r = sub.add_parser('rollback')
r.add_argument('version')
sub.add_parser('status')
args = p.parse_args()
registry = ModelRegistry(args.store)
if args.command == 'acquire':
    result = acquire_csv(args.source, args.store, expected_sha256=args.sha256)
elif args.command == 'train':
    records = load_snapshot(args.store, args.dataset_sha256)
    holdout = create_frozen_holdout(records, args.store)
    model, report = train_and_evaluate(records, holdout, seed=args.seed)
    result = registry.register(model, report, dataset_sha256=args.dataset_sha256)
    if args.promote:
        result['promotion'] = registry.promote(result['version'], max_mae=args.max_mae,
            min_r2=args.min_r2, max_mae_regression=args.max_mae_regression)
elif args.command == 'promote':
    result = registry.promote(args.version, max_mae=args.max_mae,
        min_r2=args.min_r2, max_mae_regression=args.max_mae_regression)
elif args.command == 'infer':
    result = registry.infer(args.smiles)
elif args.command == 'rollback':
    result = registry.rollback(args.version)
else:
    result = registry.accepted() or {'status': 'NO_ACCEPTED_MODEL'}
print(json.dumps(result, indent=2))
