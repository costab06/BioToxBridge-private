import csv
import hashlib
import json
import numpy as np
import pytest
from biotoxbridge.solubility.lifecycle import (acquire_csv, load_snapshot,
    create_frozen_holdout, train_and_evaluate, ModelRegistry)


def synthetic_csv(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    smiles = [f'{'C' * n}{suffix}' for n in range(1, 13) for suffix in ('O', 'N', 'Cl')]
    path = tmp_path / 'esol.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['smiles', 'measured log solubility in mols per litre'])
        writer.writeheader()
        for s in smiles:
            writer.writerow({'smiles': s, 'measured log solubility in mols per litre':
                             -0.05 * Descriptors.MolWt(Chem.MolFromSmiles(s))})
    return path


def test_acquire_snapshot_and_checksum(tmp_path):
    path = synthetic_csv(tmp_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = acquire_csv(str(path), tmp_path / 'store', expected_sha256=sha)
    assert meta['count'] == 36
    assert len(load_snapshot(tmp_path / 'store', sha)) == 36
    with pytest.raises(ValueError, match='checksum'):
        acquire_csv(str(path), tmp_path / 'store', expected_sha256='0' * 64)


def test_holdout_immutable(tmp_path):
    path = synthetic_csv(tmp_path)
    sha = acquire_csv(str(path), tmp_path / 'store')['sha256']
    rows = load_snapshot(tmp_path / 'store', sha)
    a = create_frozen_holdout(rows, tmp_path / 'store', seed=42)
    b = create_frozen_holdout(rows, tmp_path / 'store', seed=123)
    assert a == b
    assert not {r['smiles'] for r in a['test']}.isdisjoint({r['smiles'] for r in rows})


def test_registry_promotion_rejection_and_rollback(tmp_path):
    path = synthetic_csv(tmp_path)
    store = tmp_path / 'store'
    sha = acquire_csv(str(path), store)['sha256']
    rows = load_snapshot(store, sha)
    holdout = create_frozen_holdout(rows, store)
    model, report = train_and_evaluate(rows, holdout)
    registry = ModelRegistry(store)
    first = registry.register(model, report, dataset_sha256=sha)
    assert registry.promote(first['version'], max_mae=100, min_r2=-100)['accepted']
    assert registry.infer('CCO')['model_version'] == first['version']
    bad = registry.register(model, {**report, 'metrics': {'mae': 1000., 'rmse': 1000., 'r2': -100.}}, dataset_sha256=sha)
    assert not registry.promote(bad['version'])['accepted']
    assert registry.accepted()['version'] == first['version']
    registry.rollback(first['version'])
    assert registry.accepted()['version'] == first['version']


def test_no_model_inference(tmp_path):
    with pytest.raises(RuntimeError, match='No accepted model'):
        ModelRegistry(tmp_path).infer('CCO')
