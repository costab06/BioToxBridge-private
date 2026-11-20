#!/usr/bin/env python3
"""Durable per-chemical OAF artifact runner; replay-safe by stable request IDs.

Uses the existing OAF immutable artifact contract. This is a local runner,
not the Redis Streams coordinator or a recurring scheduler.
"""
import argparse
import hashlib
import json
from pathlib import Path
from biotoxbridge.oaf.artifacts import publish_json, read_json
from biotoxbridge.oaf.esol_workflows import accepted_version, lifecycle_stages, compact_status

p = argparse.ArgumentParser()
p.add_argument('--store', default='runs/esol-lifecycle')
p.add_argument('--artifact-root', default='runs/oaf-artifacts')
p.add_argument('--run-id', required=True)
p.add_argument('--input', required=True, help='JSON array of {id,smiles} objects')
p.add_argument('--tox21-url', default='http://localhost:8080')
p.add_argument('--skip-tox21', action='store_true')
a = p.parse_args()
rows = json.loads(Path(a.input).read_text())
if not isinstance(rows, list):
    raise SystemExit('Input must be a JSON array')
version = accepted_version(a.store)
root = Path(a.artifact_root)
manifest_path = root / a.run_id / 'esol-batch-manifest.json'
manifest = {'model_version': version, 'input_sha256': hashlib.sha256(
    json.dumps(rows, sort_keys=True).encode()).hexdigest(), 'include_tox21': not a.skip_tox21}
if manifest_path.exists():
    prior = json.loads(manifest_path.read_text())
    if prior['input_sha256'] != manifest['input_sha256'] or prior['include_tox21'] != manifest['include_tox21']:
        raise SystemExit('Cannot resume with different inputs or options')
    version = prior['model_version']
else:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x') as f:
        json.dump(manifest, f, indent=2)
stage = lifecycle_stages(a.store, tox21_url=a.tox21_url)['esol_infer']
results = []
seen = set()
for row in rows:
    ident = str(row['id'])
    if not ident or ident in seen:
        raise SystemExit('Every chemical must have a unique nonempty id')
    seen.add(ident)
    digest = hashlib.sha256(ident.encode()).hexdigest()[:32]
    name = f'esol-infer-{digest}'
    artifact_path = root / a.run_id / (name + '.json')
    if artifact_path.exists():
        result = json.loads(artifact_path.read_text())
    else:
        try:
            result = {'id': ident, 'status': 'ok', 'output': stage({
                'smiles': row['smiles'], 'model_version': version,
                'include_tox21': not a.skip_tox21})}
        except (ValueError, KeyError) as exc:
            result = {'id': ident, 'status': 'invalid_input', 'error': str(exc)}
        # Transient inference/service failures are not checkpointed; replay retries them.
        publish_json(root, a.run_id, name, result)
    results.append(result)
print(json.dumps({'run_id': a.run_id, 'model_version': version,
                  'completed': len(results), 'results': results}, indent=2))
