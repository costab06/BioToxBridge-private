"""Standalone ESOL acquisition, frozen evaluation, and safe model lifecycle.

No OAF dependency. All persisted models are trusted local artifacts: joblib/pickle
must never be loaded from untrusted directories.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import uuid

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from biotoxbridge.solubility.core import DESCRIPTORS, describe, predict, acceptance

ESOL_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
SOURCE_SMILES = 'smiles'
SOURCE_TARGET = 'measured log solubility in mols per litre'


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.tmp-', delete=False) as f:
        temp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)



def consolidate_records(rows: list[dict]) -> tuple[list[dict], dict]:
    """Group source rows by canonical structure before splitting.

    Multiple measured labels for the same structure are averaged, and their
    disagreement is recorded rather than silently selecting one measurement.
    """
    groups: dict[str, list[float]] = {}
    for i, row in enumerate(rows):
        canonical, _ = describe(str(row['smiles']))
        target = float(row['measured_log_solubility'])
        if not np.isfinite(target):
            raise ValueError(f'Nonfinite solubility label at row {i}')
        groups.setdefault(canonical, []).append(target)
    if len(groups) < 20:
        raise ValueError('At least 20 unique labeled structures required')
    records = []
    for canonical, values in groups.items():
        _, features = describe(canonical)
        if not np.all(np.isfinite(features)):
            raise ValueError(f'Nonfinite descriptors: {canonical}')
        records.append({'smiles': canonical, 'features': features,
                        'target': float(np.mean(values))})
    duplicates = {s: v for s, v in groups.items() if len(v) > 1}
    stats = {'source_rows': len(rows), 'unique_structures': len(groups),
             'duplicate_rows': len(rows) - len(groups),
             'duplicate_structures': len(duplicates),
             'conflicting_label_structures': sum(
                 max(v) - min(v) > 1e-6 for v in duplicates.values()),
             'max_duplicate_label_spread': float(max(
                 (max(v) - min(v) for v in duplicates.values()), default=0.0)),
             'duplicate_policy': 'mean measured log-solubility per canonical SMILES'}
    return records, stats


def acquire_csv(source: str, store: str | Path, *, expected_sha256: str | None = None) -> dict:
    """Acquire bytes from a local CSV or HTTPS URL, preserve immutable snapshot."""
    if source.startswith('https://'):
        with urllib.request.urlopen(source, timeout=60) as response:
            data = response.read(10_000_001)
    elif '://' in source:
        raise ValueError('Only HTTPS and local file sources are supported')
    else:
        data = Path(source).read_bytes()
    if len(data) > 10_000_000:
        raise ValueError('ESOL input exceeds 10 MB')
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 and sha.lower() != expected_sha256.lower():
        raise ValueError(f'Dataset checksum mismatch: {sha}')
    decoded = data.decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(decoded))
    if not {SOURCE_SMILES, SOURCE_TARGET}.issubset(reader.fieldnames or []):
        raise ValueError(f'Missing ESOL CSV fields: {SOURCE_SMILES}, {SOURCE_TARGET}')
    rows = []
    for i, row in enumerate(reader, 2):
        try:
            rows.append({'smiles': row[SOURCE_SMILES],
                         'measured_log_solubility': float(row[SOURCE_TARGET])})
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid ESOL label at CSV line {i}') from exc
    prepared, duplicate_stats = consolidate_records(rows)
    root = Path(store) / 'datasets' / sha
    csv_path = root / 'source.csv'
    metadata = {'sha256': sha, 'source': source, 'count': len(prepared),
                'target': SOURCE_TARGET, 'target_units': 'log10(mol/L)',
                'smiles_field': SOURCE_SMILES, 'deduplication': duplicate_stats}
    if csv_path.exists() and csv_path.read_bytes() != data:
        raise RuntimeError('Immutable snapshot collision')
    if not csv_path.exists():
        _atomic_write(csv_path, data)
    _atomic_write(root / 'metadata.json', _json_bytes(metadata))
    return metadata


def load_snapshot(store: str | Path, sha: str) -> list[dict]:
    raw = (Path(store) / 'datasets' / sha / 'source.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Dataset snapshot checksum mismatch')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    records, _ = consolidate_records([
        {'smiles': r[SOURCE_SMILES], 'measured_log_solubility': r[SOURCE_TARGET]}
        for r in reader])
    return records


def create_frozen_holdout(records: list[dict], store: str | Path, *, fraction=.2, seed=42) -> dict:
    """One immutable test set, indexed by canonical SMILES and label.

    Future retraining can add training data, but must never train on these structures.
    """
    from sklearn.model_selection import train_test_split
    if not 0 < fraction < .5:
        raise ValueError('Holdout fraction must be between 0 and 0.5')
    root = Path(store)
    path = root / 'evaluation' / 'frozen_holdout.json'
    if path.exists():
        return json.loads(path.read_text())
    ordered = sorted(records, key=lambda r: r['smiles'])
    indices = train_test_split(np.arange(len(ordered)), test_size=fraction, random_state=seed)[1]
    test = [{'smiles': ordered[int(i)]['smiles'], 'target': ordered[int(i)]['target']}
            for i in sorted(indices)]
    value = {'seed': seed, 'fraction': fraction, 'test': test,
             'note': 'Never use these structures or labels for training/model selection'}
    # Exclusive create prevents a concurrent retraining process replacing the frozen holdout.
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as f:
            f.write(_json_bytes(value))
    except FileExistsError:
        return json.loads(path.read_text())
    return value


def train_and_evaluate(records: list[dict], holdout: dict, *, seed=42) -> tuple[object, dict]:
    labels = {r['smiles']: r for r in records}
    test_rows = holdout['test']
    held_out = {r['smiles'] for r in test_rows}
    train_rows = [r for r in records if r['smiles'] not in held_out]
    if len(train_rows) < 20 or len(test_rows) < 2:
        raise ValueError('Insufficient train/evaluation data')
    x_train = np.asarray([r['features'] for r in train_rows], dtype=float)
    y_train = np.asarray([r['target'] for r in train_rows], dtype=float)
    # The holdout is immutable: use the original frozen labels, not a changed source label.
    from biotoxbridge.solubility.core import describe
    x_test = np.asarray([describe(r['smiles'])[1] for r in test_rows], dtype=float)
    y_test = np.asarray([r['target'] for r in test_rows], dtype=float)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    report = {'metrics': {'mae': float(mean_absolute_error(y_test, preds)),
                          'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
                          'r2': float(r2_score(y_test, preds))},
              'train_count': len(train_rows), 'test_count': len(test_rows),
              'test_smiles': [r['smiles'] for r in test_rows],
              'descriptors': list(DESCRIPTORS), 'seed': seed}
    return model, report


class ModelRegistry:
    """Immutable candidates and an atomic accepted-model pointer.

    Caller must serialize concurrent promotions (e.g. OAF coordinator lock).
    """
    def __init__(self, store: str | Path):
        self.root = Path(store) / 'models'

    def accepted(self) -> dict | None:
        pointer = self.root / 'accepted.json'
        if not pointer.exists():
            return None
        value = json.loads(pointer.read_text())
        return self.candidate(value['version'])

    def candidate(self, version: str) -> dict:
        if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in version):
            raise ValueError('Invalid model version')
        return json.loads((self.root / 'versions' / version / 'report.json').read_text())

    def register(self, model, report: dict, *, dataset_sha256: str) -> dict:
        version = uuid.uuid4().hex
        root = self.root / 'versions' / version
        root.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, root / 'model.joblib')
        model_hash = hashlib.sha256((root / 'model.joblib').read_bytes()).hexdigest()
        record = {**report, 'version': version, 'dataset_sha256': dataset_sha256,
                  'model_sha256': model_hash, 'model_file': 'model.joblib'}
        _atomic_write(root / 'report.json', _json_bytes(record))
        return record

    def promote(self, version: str, *, max_mae=1.0, min_r2=0.2, max_mae_regression=0.0) -> dict:
        candidate = self.candidate(version)
        incumbent = self.accepted()
        decision = acceptance(candidate, incumbent, max_mae=max_mae, min_r2=min_r2,
                              max_mae_regression=max_mae_regression)
        if decision['accepted']:
            prior = incumbent['version'] if incumbent else None
            history = self.root / 'promotion_history.jsonl'
            history.parent.mkdir(parents=True, exist_ok=True)
            # Append history before changing the pointer so the previous version is recorded.
            with history.open('a') as f:
                f.write(json.dumps({'version': version, 'previous': prior, 'decision': decision}) + '\n')
                f.flush()
                os.fsync(f.fileno())
            _atomic_write(self.root / 'accepted.json', _json_bytes({'version': version}))
        return decision

    def rollback(self, version: str) -> dict:
        self.candidate(version)  # validate existing version
        current = self.accepted()
        with (self.root / 'promotion_history.jsonl').open('a') as f:
            f.write(json.dumps({'rollback_to': version, 'previous': current['version'] if current else None}) + '\n')
            f.flush()
            os.fsync(f.fileno())
        _atomic_write(self.root / 'accepted.json', _json_bytes({'version': version}))
        return self.candidate(version)

    def infer(self, smiles: str) -> dict:
        record = self.accepted()
        if record is None:
            raise RuntimeError('No accepted model; train and promote one first')
        path = self.root / 'versions' / record['version'] / record['model_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['model_sha256']:
            raise ValueError('Accepted model artifact checksum mismatch')
        model = joblib.load(path)  # trusted local model files only
        return {**predict(model, smiles), 'model_version': record['version']}
