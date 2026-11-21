import pytest
from biotoxbridge.solubility.core import describe, prepare, acceptance


def test_canonical_descriptors():
    a, x = describe('CCO')
    b, y = describe('OCC')
    import numpy as np

    assert a == b
    assert len(x) == len(y) == 7
    np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_invalid_smiles():
    with pytest.raises(ValueError, match='Invalid SMILES'):
        describe('not-a-smiles')


def test_duplicate_structure_rejected():
    rows = [{'smiles': 'CCO', 'measured_log_solubility': -1},
            {'smiles': 'OCC', 'measured_log_solubility': -1}]
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(rows)


def test_acceptance_rejects_regression():
    old = {'metrics': {'mae': .4}, 'test_smiles': ['CCO']}
    new = {'metrics': {'mae': .5, 'r2': .5}, 'test_smiles': ['CCO']}
    assert acceptance(new, old)['accepted'] is False


def test_acceptance_rejects_different_holdout():
    old = {'metrics': {'mae': .4}, 'test_smiles': ['CCO']}
    new = {'metrics': {'mae': .3, 'r2': .5}, 'test_smiles': ['CCC']}
    assert 'evaluation_set_mismatch' in acceptance(new, old)['reasons']
