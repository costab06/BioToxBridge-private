"""Compare OAF preparation artifacts with a fresh direct scientific-stage run.

Run from the BioToxBridge repository root with its normal dependencies/services.
No OAF imports or Redis connection are needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import yaml

from biotoxbridge.oaf.artifacts import read_json
from biotoxbridge.oaf.scientific import scientific_stages

STAGES = (
    'toxicity_target', 'chemistry_resolution', 'structure_grouping',
    'tox21_scoring', 'master_assembly', 'dataset_split',
)


def canonical(value):
    """Canonicalize JSON structures, ignoring ordering of lists of records."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        items = [canonical(v) for v in value]
        # Only reorder lists of records; leave other lists (e.g. numeric features) intact.
        if all(isinstance(v, dict) for v in items):
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, allow_nan=False))
        return items
    return value


# Only optional chemistry metadata on records explicitly rejected for missing
# QSAR-ready SMILES. DTXSID, accepted structures, and predictions stay strict.
OPTIONAL_MISSING_METADATA = frozenset({'inchiString', 'inchikey', 'msReadySmiles', 'dtxcid', 'smiles'})


def normalize_missing_metadata(value):
    """Return normalized value and number of changed metadata fields."""
    if isinstance(value, list):
        items = [normalize_missing_metadata(v) for v in value]
        return [v for v, _ in items], sum(n for _, n in items)
    if isinstance(value, dict):
        out = {}
        count = 0
        eligible = value.get('failure_reason') == 'qsarReadySmiles_missing'
        for key, child in value.items():
            if eligible and key in OPTIONAL_MISSING_METADATA and child in ('', None):
                out[key] = None
                count += int(child == '')
            else:
                out[key], n = normalize_missing_metadata(child)
                count += n
        return out, count
    return value, 0


def fingerprint(value):
    payload = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def oaf_stage_payload(root: Path, run_id: str, stage: str):
    paths = sorted((root / run_id).glob(f'{stage}-*.json'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one OAF artifact for {stage}; found {len(paths)}: {paths}')
    path = paths[0]
    content = path.read_bytes()
    reference = {
        'artifact_id': f'{run_id}/{path.name}',
        'uri': str(path.resolve()),
        'checksum': hashlib.sha256(content).hexdigest(),
    }
    return read_json(reference, root)


def compare_stage(direct, oaf):
    keys = sorted(set(direct) | set(oaf))
    differences = []
    normalized_fields = []
    for key in keys:
        if key not in direct or key not in oaf:
            differences.append({'field': key, 'reason': 'missing field'})
        else:
            d_normal, d_count = normalize_missing_metadata(direct[key]) if key == "missing" else (direct[key], 0)
            o_normal, o_count = normalize_missing_metadata(oaf[key]) if key == "missing" else (oaf[key], 0)
            if fingerprint(d_normal) == fingerprint(o_normal):
                if fingerprint(direct[key]) != fingerprint(oaf[key]):
                    normalized_fields.append({"field": key, "rule": "optional missing-chemistry metadata: empty string equals null", "direct_empty_fields": d_count, "oaf_empty_fields": o_count})
                continue
            d, o = d_normal, o_normal
            item = {'field': key, 'reason': 'content differs',
                    'direct_sha256': fingerprint(d), 'oaf_sha256': fingerprint(o)}
            if isinstance(d, list) and isinstance(o, list):
                item['direct_count'] = len(d)
                item['oaf_count'] = len(o)
                dc = Counter(json.dumps(v, sort_keys=True, allow_nan=False) for v in d)
                oc = Counter(json.dumps(v, sort_keys=True, allow_nan=False) for v in o)
                item['direct_only_samples'] = list((dc - oc).elements())[:2]
                item['oaf_only_samples'] = list((oc - dc).elements())[:2]
            differences.append(item)
    return differences, normalized_fields


def verify(config_path, run_id, artifact_root, direct_work_dir, report_path, cache_csv=None):
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping')
    if direct_work_dir.exists() and any(direct_work_dir.iterdir()):
        raise ValueError(f'Direct output directory must be empty: {direct_work_dir}')
    direct_work_dir.mkdir(parents=True, exist_ok=True)
    interim = None
    if cache_csv:
        if not cache_csv.is_file():
            raise FileNotFoundError(cache_csv)
        interim = direct_work_dir / 'interim'
        interim.mkdir(exist_ok=True)
        (interim / 'dsstox_cache.csv').write_bytes(cache_csv.read_bytes())
    stages = scientific_stages(config, direct_work_dir, interim_dir=interim)
    state = {}
    results = []
    for stage in STAGES:
        print(f'Running direct stage: {stage}', flush=True)
        state = stages[stage](state)
        oaf = oaf_stage_payload(artifact_root, run_id, stage)
        differences, normalized_fields = compare_stage(state, oaf)
        results.append({'stage': stage, 'match': not differences,
                        'differences': differences,
                        'normalizations': normalized_fields})
        print(f'  {"MATCH" if not differences else "DIFFERENT"}', flush=True)
    report = {'run_id': run_id, 'config': str(config_path),
              'artifact_root': str(artifact_root),
              'all_match': all(r['match'] for r in results), 'stages': results}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Comparison report: {report_path}')
    print('PASS: all stages equivalent' if report['all_match'] else 'FAIL: differences detected')
    return 0 if report['all_match'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--artifact-root', type=Path, default=Path('runs/oaf-artifacts'))
    parser.add_argument('--direct-work-dir', type=Path, default=Path('runs/equivalence-direct'))
    parser.add_argument('--report', type=Path, default=Path('runs/equivalence-report.json'))
    parser.add_argument('--cache-csv', type=Path, help='Optional known DSSTox cache to avoid EPA calls')
    args = parser.parse_args()
    try:
        return verify(args.config, args.run_id, args.artifact_root,
                      args.direct_work_dir, args.report, args.cache_csv)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
