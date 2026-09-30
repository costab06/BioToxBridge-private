"""Unit tests for features.fingerprints."""

import numpy as np
import pytest

from biotoxbridge.features.fingerprints import morgan_fp, fp_matrix


def test_morgan_fp_shape():
    arr = morgan_fp("CCO")
    assert arr.shape == (2048,)
    assert arr.dtype == np.uint8


def test_morgan_fp_binary():
    arr = morgan_fp("CCO")
    assert set(arr.tolist()).issubset({0, 1})


def test_morgan_fp_nonzero():
    arr = morgan_fp("CCO")
    assert arr.sum() > 0


def test_morgan_fp_different_smiles():
    arr1 = morgan_fp("CCO")
    arr2 = morgan_fp("c1ccccc1")   # benzene
    assert not np.array_equal(arr1, arr2)


def test_morgan_fp_same_smiles():
    """Canonical SMILES should give same FP regardless of input order."""
    arr1 = morgan_fp("CCO")
    arr2 = morgan_fp("OCC")   # same molecule, different order
    assert np.array_equal(arr1, arr2)


def test_morgan_fp_invalid_smiles():
    with pytest.raises(ValueError):
        morgan_fp("NOT_VALID!!!")


def test_fp_matrix_shape():
    recs = [
        {"dtxsid": "DTXSID1", "canonical_smiles": "CCO"},
        {"dtxsid": "DTXSID2", "canonical_smiles": "CCCO"},
    ]
    X, ids = fp_matrix(recs)
    assert X.shape == (2, 2048)
    assert X.dtype == np.float32
    assert ids == ["DTXSID1", "DTXSID2"]


def test_fp_matrix_custom_bits():
    recs = [{"dtxsid": "D1", "canonical_smiles": "c1ccccc1"}]
    X, _ = fp_matrix(recs, n_bits=1024)
    assert X.shape == (1, 1024)
