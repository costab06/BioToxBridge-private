"""Unit tests for evaluation.evaluator."""

import math
import numpy as np
import pytest

from biotoxbridge.evaluation.evaluator import evaluate, bootstrap_ci, compare_models


def _perfect(n=20):
    y = np.linspace(1.0, 3.0, n).astype(np.float32)
    return y, y.copy()


def _constant_pred(n=20):
    y = np.linspace(1.0, 3.0, n).astype(np.float32)
    return y, np.full(n, 2.0, dtype=np.float32)


def test_perfect_predictions():
    y, yp = _perfect()
    m = evaluate(y, yp, label="test")
    assert m["mae"]  == pytest.approx(0.0, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["r2"]   == pytest.approx(1.0, abs=1e-6)
    assert m["spearman_rho"] == pytest.approx(1.0, abs=1e-6)


def test_constant_prediction_r2_negative_or_zero():
    y, yp = _constant_pred()
    m = evaluate(y, yp)
    assert m["r2"] <= 0.0


def test_evaluate_n():
    y, yp = _perfect(15)
    assert evaluate(y, yp)["n"] == 15


def test_bootstrap_ci_keys():
    y, yp = _constant_pred()
    ci = bootstrap_ci(y, yp, n_iter=50, seed=0)
    for key in ("mae", "rmse", "r2", "spearman_rho"):
        assert key in ci
        assert "mean" in ci[key]
        assert "ci_lo" in ci[key]
        assert "ci_hi" in ci[key]


def test_bootstrap_ci_lo_le_hi():
    # Use a non-constant prediction so Spearman is defined in bootstrap samples
    rng = np.random.default_rng(0)
    y  = np.linspace(1.0, 3.0, 30).astype(np.float32)
    yp = y + rng.normal(0, 0.3, size=30).astype(np.float32)
    ci = bootstrap_ci(y, yp, n_iter=100, seed=0)
    for key in ci:
        assert ci[key]["ci_lo"] <= ci[key]["ci_hi"]


def test_compare_models():
    y, yp = _perfect()
    m1 = evaluate(y, yp, label="A")
    m2 = evaluate(y, np.zeros_like(y), label="B")
    table = compare_models({"A": m1, "B": m2})
    assert "mae" in table
    assert table["mae"]["A"] == pytest.approx(0.0, abs=1e-6)
    assert table["mae"]["B"] > 0
