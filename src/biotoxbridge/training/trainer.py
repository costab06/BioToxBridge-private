"""Ridge regression trainer for Models A / B / D.

Fits a Ridge regressor with cross-validated alpha selection on the training
partition.  Returns the fitted model, best alpha, and CV scores.

For the A/D models (high-dimensional fingerprints) we also optionally apply
a truncated SVD to reduce dimensionality before regression — this is kept
disabled by default so the baseline is a plain Ridge on raw FP bits.

Usage:
    from biotoxbridge.training.trainer import train_model

    model, info = train_model(X_train, y_train, model_id="A")
"""

from __future__ import annotations

import sys
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_id: str = "?",
    alphas: tuple[float, ...] | None = None,
    cv: int = 5,
) -> tuple[Pipeline, dict]:
    """
    Fit a RidgeCV on (X_train, y_train).

    Parameters
    ----------
    X_train   : float32 ndarray shape (n_train, n_features)
    y_train   : float32 ndarray shape (n_train,) — already scaled by projector
    model_id  : label for logging ("A", "B", or "D")
    alphas    : iterable of alpha values to search (default: log-spaced 1e-3..1e4)
    cv        : number of CV folds (default: 5)

    Returns
    -------
    (pipeline, info)
      pipeline : fitted sklearn Pipeline (just a RidgeCV estimator)
      info     : dict with best_alpha, cv_r2_scores, n_train, n_features
    """
    if alphas is None:
        alphas = tuple(np.logspace(-3, 4, 30))

    ridge = RidgeCV(alphas=alphas, cv=cv, scoring="r2", store_cv_results=True)
    ridge.fit(X_train, y_train)

    best_alpha = float(ridge.alpha_)
    # cv_values: shape (n_samples, n_alphas) — mean over samples for diagnostics
    best_alpha_idx = list(alphas).index(best_alpha) if best_alpha in alphas else -1

    info = {
        "model_id":   model_id,
        "best_alpha": best_alpha,
        "n_train":    X_train.shape[0],
        "n_features": X_train.shape[1],
        "cv_folds":   cv,
    }

    print(
        f"  Model {model_id}: alpha={best_alpha:.4g}  "
        f"n_train={X_train.shape[0]}  n_feat={X_train.shape[1]}",
        file=sys.stderr,
    )

    from sklearn.pipeline import Pipeline as _Pipe
    pipeline = _Pipe([("ridge", ridge)])
    return pipeline, info
