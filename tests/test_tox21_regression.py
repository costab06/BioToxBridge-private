import csv
import math
import pytest
from biotoxbridge.features.tox21_adapter import Tox21Adapter

ENDPOINTS = ['NR-AR', 'SR-p53']


def _reference(tmp_path, rows):
    path = tmp_path / 'reference.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['SMILES', 'Valid'] + [f'{ep}_score' for ep in ENDPOINTS])
        w.writeheader()
        w.writerows(rows)
    return str(path)


def _scored(smiles, a=0.5, b=-0.2):
    return {'canonical_smiles': smiles, 'tox21_NR-AR': a, 'tox21_SR-p53': b}


def test_pass_matches_canonical_smiles(tmp_path):
    ref = _reference(tmp_path, [{'SMILES': 'C(C)O', 'Valid': '1', 'NR-AR_score': '0.5', 'SR-p53_score': '-0.2'}])
    report = Tox21Adapter(endpoints=ENDPOINTS).regression_test([_scored('CCO')], ref)
    assert report['status'] == 'PASS'
    assert report['n_matched'] == 1
    assert report['coverage'] == 1


def test_fail_when_tolerance_exceeded(tmp_path):
    ref = _reference(tmp_path, [{'SMILES': 'CCO', 'Valid': '1', 'NR-AR_score': '0.8', 'SR-p53_score': '-0.2'}])
    report = Tox21Adapter(endpoints=ENDPOINTS).regression_test([_scored('CCO')], ref)
    assert report['status'] == 'FAIL'


def test_not_comparable_when_no_matches(tmp_path):
    ref = _reference(tmp_path, [{'SMILES': 'CCO', 'Valid': '1', 'NR-AR_score': '0.5', 'SR-p53_score': '-0.2'}])
    report = Tox21Adapter(endpoints=ENDPOINTS).regression_test([_scored('O')], ref)
    assert report['status'] == 'NOT_COMPARABLE'
    assert report['n_unmatched'] == 1


def test_duplicate_reference_rejected(tmp_path):
    row = {'SMILES': 'CCO', 'Valid': '1', 'NR-AR_score': '0.5', 'SR-p53_score': '-0.2'}
    ref = _reference(tmp_path, [row, {**row, 'SMILES': 'C(C)O'}])
    with pytest.raises(ValueError, match='Duplicate reference'):
        Tox21Adapter(endpoints=ENDPOINTS).regression_test([_scored('CCO')], ref)


def test_nonfinite_reference_rejected(tmp_path):
    ref = _reference(tmp_path, [{'SMILES': 'CCO', 'Valid': '1', 'NR-AR_score': 'nan', 'SR-p53_score': '-0.2'}])
    with pytest.raises(ValueError, match='Non-finite'):
        Tox21Adapter(endpoints=ENDPOINTS).regression_test([_scored('CCO')], ref)


def test_partial_coverage_reported(tmp_path):
    ref = _reference(tmp_path, [{'SMILES': 'CCO', 'Valid': '1', 'NR-AR_score': '0.5', 'SR-p53_score': '-0.2'}])
    report = Tox21Adapter(endpoints=ENDPOINTS).regression_test([_scored('CCO'), _scored('O')], ref)
    assert report['status'] == 'PASS'
    assert report['coverage'] == 0.5
