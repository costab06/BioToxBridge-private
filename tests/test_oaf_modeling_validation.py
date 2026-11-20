import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'validate_oaf_modeling.py'
spec = importlib.util.spec_from_file_location('validate_oaf_modeling', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_paired_bootstrap_identical_models():
    rows = {}
    for i in range(12):
        for model in ('A', 'D'):
            rows[(model, str(i))] = (float(i), float(i) + (i % 3) * .1)
    result = m.paired_bootstrap(rows, iterations=100, seed=3)
    assert result['delta_mae']['ci_lo'] == pytest.approx(0)
    assert result['delta_rmse']['ci_hi'] == pytest.approx(0)


def test_mismatched_ids_rejected():
    with pytest.raises(ValueError, match='identical'):
        m.paired_bootstrap({('A','1'): (1.,1.), ('D','2'): (2.,2.)}, iterations=10)


def test_comparison_requires_matching_predictions():
    left = {('A','1'): (1., 1.2)}
    assert m.compare_predictions(left, dict(left))['pass']
    assert not m.compare_predictions(left, {('A','1'): (1., 1.3)})['pass']
