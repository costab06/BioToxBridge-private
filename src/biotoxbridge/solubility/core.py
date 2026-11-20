"""Deterministic molecular descriptors and solubility model training.

This module has no OAF dependency. Model promotion is intentionally separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DESCRIPTORS = ('MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount')


def describe(smiles: str) -> tuple[str, list[float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    canonical = Chem.MolToSmiles(mol, canonical=True)
    values = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
              rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
              rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol)]
    return canonical, [float(x) for x in values]


def prepare(rows: list[dict], *, smiles_key='smiles', target_key='measured_log_solubility') -> list[dict]:
    """Validate labels and structures; reject duplicates to avoid train/test leakage."""
    result, seen = [], set()
    for index, row in enumerate(rows):
        canonical, features = describe(str(row[smiles_key]))
        y = float(row[target_key])
        if not np.isfinite(y) or not np.all(np.isfinite(features)):
            raise ValueError(f'Nonfinite row {index}')
        if canonical in seen:
            raise ValueError(f'Duplicate canonical structure: {canonical}')
        seen.add(canonical)
        result.append({'smiles': canonical, 'features': features, 'target': y})
    if len(result) < 20:
        raise ValueError('At least 20 unique labeled structures required')
    return result


def train_candidate(rows: list[dict], *, seed=42, test_fraction=0.2) -> tuple[object, dict]:
    """Train on one split; report held-out metrics. No model promotion."""
    records = prepare(rows)
    x = np.asarray([r['features'] for r in records], dtype=float)
    y = np.asarray([r['target'] for r in records], dtype=float)
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(indices, test_size=test_fraction, random_state=seed)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    metrics = {'mae': float(mean_absolute_error(y[test_idx], pred)),
               'rmse': float(np.sqrt(mean_squared_error(y[test_idx], pred))),
               'r2': float(r2_score(y[test_idx], pred))}
    report = {'dataset_sha256': digest, 'seed': seed, 'train_count': len(train_idx),
              'test_count': len(test_idx), 'descriptors': list(DESCRIPTORS), 'metrics': metrics,
              'test_smiles': [records[i]['smiles'] for i in test_idx],
              'test_targets': y[test_idx].tolist(), 'test_predictions': pred.tolist()}
    return model, report


def predict(model, smiles: str) -> dict:
    canonical, values = describe(smiles)
    prediction = float(model.predict(np.asarray([values], dtype=float))[0])
    return {'canonical_smiles': canonical, 'predicted_log_solubility': prediction}


def acceptance(candidate: dict, incumbent: dict | None, *, max_mae=1.0, min_r2=0.2, max_mae_regression=0.0) -> dict:
    """Example policy; compare on the same frozen evaluation set in production."""
    m = candidate['metrics']
    reasons = []
    if m['mae'] > max_mae: reasons.append('mae_above_threshold')
    if m['r2'] < min_r2: reasons.append('r2_below_threshold')
    if incumbent is not None:
        if candidate.get('test_smiles') != incumbent.get('test_smiles'):
            reasons.append('evaluation_set_mismatch')
        elif m['mae'] > incumbent['metrics']['mae'] + max_mae_regression:
            reasons.append('mae_regression')
    return {'accepted': not reasons, 'reasons': reasons}
