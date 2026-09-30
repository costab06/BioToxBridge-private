"""Evaluation metrics and bootstrap confidence intervals.

Computes MAE, RMSE, R², Spearman rho on the held-out test partition and
optionally produces bootstrap CIs.

Usage:
    from biotoxbridge.evaluation.evaluator import evaluate, bootstrap_ci, compare_models

    metrics = evaluate(y_true, y_pred, label="Model A")
    ci      = bootstrap_ci(y_true, y_pred, n_iter=1000, seed=42)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
    y_scaler=None,
) -> dict:
    """
    Compute MAE, RMSE, R², Spearman rho.

    If y_scaler is supplied, metrics are computed in the original
    log10(mg/kg/day) space rather than the scaled space.
    """
    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    residuals = y_true - y_pred
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho, pval = stats.spearmanr(y_true, y_pred)

    result = {
        "label":   label,
        "n":       len(y_true),
        "mae":     mae,
        "rmse":    rmse,
        "r2":      r2,
        "spearman_rho": float(rho),
        "spearman_p":   float(pval),
    }
    return result


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    y_scaler=None,
) -> dict:
    """
    Bootstrap confidence intervals (percentile method) for all metrics.

    Returns dict: metric → {"mean": ..., "ci_lo": ..., "ci_hi": ...}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {k: [] for k in ("mae", "rmse", "r2", "spearman_rho")}

    for _ in range(n_iter):
        idx  = rng.integers(0, n, size=n)
        m    = evaluate(y_true[idx], y_pred[idx], y_scaler=y_scaler)
        for k in samples:
            samples[k].append(m[k])

    ci = {}
    lo, hi = alpha / 2, 1 - alpha / 2
    for k, vals in samples.items():
        arr = np.array(vals)
        ci[k] = {
            "mean":  float(np.mean(arr)),
            "ci_lo": float(np.nanpercentile(arr, lo * 100)),
            "ci_hi": float(np.nanpercentile(arr, hi * 100)),
        }
    return ci


def compare_models(
    results: dict[str, dict],
    metrics: tuple[str, ...] = ("mae", "rmse", "r2", "spearman_rho"),
) -> dict:
    """
    Produce a comparison table across models.

    Parameters
    ----------
    results : {model_label: evaluate() output dict}

    Returns
    -------
    dict with keys = metric names, values = {model_label: value}
    """
    table: dict[str, dict[str, float]] = {m: {} for m in metrics}
    for label, res in results.items():
        for m in metrics:
            table[m][label] = res.get(m, float("nan"))
    return table
