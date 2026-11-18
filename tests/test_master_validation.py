"""Contract tests for numerical Tox21 validation during master assembly."""
import math
import pytest
from biotoxbridge.datasets.master import build_master
from biotoxbridge.features.tox21_adapter import TOX21_SCORE_FIELDS


def records(overrides=None):
    target = [{"dtxsid": "DTXSID1", "preferred_name": "sample"}]
    scored = {"dtxsid": "DTXSID1", "canonical_smiles": "CCO",
              "structure_group": 0,
              **{f: "0.25" for f in TOX21_SCORE_FIELDS}}
    scored.update(overrides or {})
    return target, [scored]


def test_valid_scores_converted_to_floats():
    master, excluded = build_master(*records())
    assert not excluded
    assert len(master) == 1
    assert all(isinstance(master[0][f], float) for f in TOX21_SCORE_FIELDS)


def test_negative_and_above_one_scores_are_allowed():
    a, b = TOX21_SCORE_FIELDS[:2]
    master, excluded = build_master(*records({a: "-0.108", b: "1.076"}))
    assert not excluded
    assert master[0][a] == pytest.approx(-0.108)
    assert master[0][b] == pytest.approx(1.076)


@pytest.mark.parametrize('bad', ['oops', 'NaN', 'inf', '-inf', float('nan'), float('inf'), {}, []])
def test_invalid_score_is_excluded_and_audited(bad):
    field = TOX21_SCORE_FIELDS[0]
    master, excluded = build_master(*records({field: bad}))
    assert master == []
    assert len(excluded) == 1
    assert excluded[0]['exclusion_reason'] == 'invalid_tox21_scores'
    assert field in excluded[0]['invalid_fields']


@pytest.mark.parametrize('missing', [None, '', '   '])
def test_missing_score_is_excluded_and_audited(missing):
    field = TOX21_SCORE_FIELDS[0]
    master, excluded = build_master(*records({field: missing}))
    assert master == []
    assert excluded[0]['exclusion_reason'] == 'missing_tox21_scores'
    assert field in excluded[0]['missing_fields']


def test_multiple_invalid_fields_all_reported():
    a, b = TOX21_SCORE_FIELDS[:2]
    master, excluded = build_master(*records({a: 'bad', b: 'inf'}))
    assert not master
    assert excluded[0]['invalid_fields'] == f'{a},{b}'


def test_invalid_exclusions_written_to_csv(tmp_path):
    field = TOX21_SCORE_FIELDS[0]
    audit = tmp_path / 'audit.csv'
    master, excluded = build_master(*records({field: 'NaN'}), audit_csv=audit)
    assert not master
    assert audit.exists()
    content = audit.read_text()
    assert 'invalid_tox21_scores' in content
    assert field in content
