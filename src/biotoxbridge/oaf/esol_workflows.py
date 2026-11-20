"""ESOL lifecycle operations exposed as OAF-compatible JSON stages.

These stages are synchronous functions suitable for OAF BaseAgent's execute_stage.
The outer scheduler/event coordinator is deliberately separate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from biotoxbridge.solubility.lifecycle import (ESOL_URL, ModelRegistry, acquire_csv,
    load_snapshot, create_frozen_holdout, train_and_evaluate)


def lifecycle_stages(store: str | Path, *, tox21_url: str = 'http://localhost:8080'):
    store = Path(store)
    registry = ModelRegistry(store)

    def acquire(request):
        return acquire_csv(request.get('source', ESOL_URL), store,
                           expected_sha256=request.get('expected_sha256'))

    def train(request):
        sha = request['dataset_sha256']
        records = load_snapshot(store, sha)
        holdout = create_frozen_holdout(records, store)
        model, report = train_and_evaluate(records, holdout, seed=request.get('seed', 42))
        record = registry.register(model, report, dataset_sha256=sha)
        return {'candidate_version': record['version'], 'metrics': record['metrics'],
                'dataset_sha256': sha, 'train_count': record['train_count'],
                'test_count': record['test_count']}

    def promote(request):
        version = request['candidate_version']
        decision = registry.promote(version, max_mae=request.get('max_mae', 1.0),
                                    min_r2=request.get('min_r2', 0.2),
                                    max_mae_regression=request.get('max_mae_regression', 0.0))
        return {'candidate_version': version, **decision,
                'accepted_version': (registry.accepted() or {}).get('version')}

    def infer(request):
        from biotoxbridge.features.tox21_adapter import Tox21Adapter
        pinned = request['model_version']
        record = registry.candidate(pinned)
        import joblib
        from biotoxbridge.solubility.core import predict
        model_path = registry.root / 'versions' / pinned / record['model_file']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != record['model_sha256']:
            raise ValueError('Pinned model checksum mismatch')
        model = joblib.load(model_path)  # trusted local registry only
        result = predict(model, request['smiles'])
        result['model_version'] = pinned
        if request.get('include_tox21', True):
            adapter = Tox21Adapter(base_url=tox21_url)
            result['tox21'] = adapter.score_smiles(result['canonical_smiles'])
        return result

    return {'esol_acquire': acquire, 'esol_train': train,
            'esol_promote': promote, 'esol_infer': infer}


def accepted_version(store):
    record = ModelRegistry(store).accepted()
    if not record:
        raise RuntimeError('No accepted ESOL model')
    return record['version']


def compact_status(store):
    registry = ModelRegistry(store)
    record = registry.accepted()
    if record is None:
        return {'status': 'NO_ACCEPTED_MODEL'}
    history = registry.root / 'promotion_history.jsonl'
    return {'accepted_version': record['version'], 'dataset_sha256': record['dataset_sha256'],
            'metrics': record['metrics'],
            'promotion_events': sum(1 for _ in history.open()) if history.exists() else 0}
