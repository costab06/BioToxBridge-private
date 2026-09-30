"""Unit tests for chemistry.structure_groups."""

import pytest
from biotoxbridge.chemistry.structure_groups import assign_groups, group_summary


BASE = {
    "dtxsid": "DTXSID0000001",
    "qsarReadySmiles": "CCO",
    "preferredName": "Ethanol",
}


def test_single_record():
    recs = [BASE.copy()]
    enriched, failed = assign_groups(recs)
    assert len(enriched) == 1
    assert len(failed) == 0
    assert "canonical_smiles" in enriched[0]
    assert "structure_group" in enriched[0]


def test_identical_smiles_same_group():
    recs = [
        {**BASE, "dtxsid": "DTXSID0000001", "qsarReadySmiles": "CCO"},
        {**BASE, "dtxsid": "DTXSID0000002", "qsarReadySmiles": "OCC"},  # same structure
    ]
    enriched, _ = assign_groups(recs)
    assert len(enriched) == 2
    groups = [r["structure_group"] for r in enriched]
    assert groups[0] == groups[1]


def test_different_smiles_different_group():
    recs = [
        {**BASE, "dtxsid": "DTXSID0000001", "qsarReadySmiles": "CCO"},
        {**BASE, "dtxsid": "DTXSID0000002", "qsarReadySmiles": "CCCO"},
    ]
    enriched, _ = assign_groups(recs)
    assert len(enriched) == 2
    groups = [r["structure_group"] for r in enriched]
    assert groups[0] != groups[1]


def test_empty_smiles_goes_to_failed():
    recs = [{**BASE, "qsarReadySmiles": ""}]
    enriched, failed = assign_groups(recs)
    assert len(enriched) == 0
    assert len(failed) == 1
    assert "failure_reason" in failed[0]


def test_invalid_smiles_goes_to_failed():
    recs = [{**BASE, "qsarReadySmiles": "NOT_A_SMILES!!!"}]
    enriched, failed = assign_groups(recs)
    assert len(enriched) == 0
    assert len(failed) == 1


def test_group_summary():
    recs = [
        {**BASE, "dtxsid": "DTXSID0000001", "qsarReadySmiles": "CCO"},
        {**BASE, "dtxsid": "DTXSID0000002", "qsarReadySmiles": "OCC"},   # dup
        {**BASE, "dtxsid": "DTXSID0000003", "qsarReadySmiles": "CCCO"},  # unique
    ]
    enriched, _ = assign_groups(recs)
    s = group_summary(enriched)
    assert s["n_chemicals"] == 3
    assert s["n_groups"] == 2
    assert s["n_singletons"] == 1
    assert s["n_duplicates"] == 1
