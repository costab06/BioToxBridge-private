"""Model A / B / D feature-matrix projectors.

Each projector takes train and test record lists (from splitter.load_split)
and returns (X_train, X_test, y_train, y_test, dtxsids_train, dtxsids_test).

Model families:
  A — 2048-bit Morgan fingerprint only
  B — 12 Tox21 continuous scores only
  D — Morgan FP + 12 Tox21 scores concatenated

Preprocessing:
  - Scaling (StandardScaler) and any other learned transforms are fit on the
    training set only and applied to both train and test.
  - Fitted scalers are returned so they can be serialised with the run.

Usage:
    from biotoxbridge.datasets.projectors import project_A, project_B, project_D

    Xa_tr, Xa_te, y_tr, y_te, ids_tr, ids_te, meta = project_A(train, test)
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from biotoxbridge.features.fingerprints import fp_matrix
from biotoxbridge.features.tox21_adapter import TOX21_SCORE_FIELDS


# ── helpers ───────────────────────────────────────────────────────────────────

def _targets(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Extract log10_loael_median targets and dtxsids."""
    y     = np.array([float(r["log10_loael_median"]) for r in records],
                     dtype=np.float32)
    ids   = [r["dtxsid"] for r in records]
    return y, ids


def _tox21_matrix(records: list[dict]) -> np.ndarray:
    return np.array(
        [[float(r[f]) for f in TOX21_SCORE_FIELDS] for r in records],
        dtype=np.float32,
    )


ProjectResult = tuple[
    np.ndarray, np.ndarray,        # X_train, X_test
    np.ndarray, np.ndarray,        # y_train, y_test
    list[str],  list[str],         # dtxsids_train, dtxsids_test
    dict,                          # meta (scaler, shape info)
]


# ── Model A: structure only ───────────────────────────────────────────────────

def project_A(
    train: list[dict],
    test:  list[dict],
    radius: int = 2,
    n_bits: int = 2048,
) -> ProjectResult:
    """2048-bit Morgan fingerprint; no scaling (binary features)."""
    X_tr, ids_tr = fp_matrix(train, radius=radius, n_bits=n_bits)
    X_te, ids_te = fp_matrix(test,  radius=radius, n_bits=n_bits)
    y_tr, _      = _targets(train)
    y_te, _      = _targets(test)

    # Scale targets only (features are already [0,1])
    y_scaler = StandardScaler()
    y_tr_sc  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te_sc  = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    meta = {
        "model":    "A",
        "n_bits":   n_bits,
        "radius":   radius,
        "y_scaler": y_scaler,
        "X_shape":  X_tr.shape,
    }
    return X_tr, X_te, y_tr_sc, y_te_sc, ids_tr, ids_te, meta


# ── Model B: predicted bioactivity only ──────────────────────────────────────

def project_B(
    train: list[dict],
    test:  list[dict],
) -> ProjectResult:
    """12 Tox21 continuous scores; StandardScaler fit on train."""
    X_tr_raw = _tox21_matrix(train)
    X_te_raw = _tox21_matrix(test)

    x_scaler = StandardScaler()
    X_tr     = x_scaler.fit_transform(X_tr_raw)
    X_te     = x_scaler.transform(X_te_raw)

    y_tr, ids_tr = _targets(train)
    y_te, ids_te = _targets(test)

    y_scaler = StandardScaler()
    y_tr_sc  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te_sc  = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    meta = {
        "model":    "B",
        "endpoints": TOX21_SCORE_FIELDS,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "X_shape":  X_tr.shape,
    }
    return X_tr, X_te, y_tr_sc, y_te_sc, ids_tr, ids_te, meta


# ── Model D: structure + predicted bioactivity ────────────────────────────────

def project_D(
    train: list[dict],
    test:  list[dict],
    radius: int = 2,
    n_bits: int = 2048,
) -> ProjectResult:
    """Morgan FP (no scale) concatenated with scaled Tox21 scores."""
    fp_tr, ids_tr = fp_matrix(train, radius=radius, n_bits=n_bits)
    fp_te, ids_te = fp_matrix(test,  radius=radius, n_bits=n_bits)

    t_tr_raw = _tox21_matrix(train)
    t_te_raw = _tox21_matrix(test)

    t_scaler = StandardScaler()
    t_tr     = t_scaler.fit_transform(t_tr_raw)
    t_te     = t_scaler.transform(t_te_raw)

    X_tr = np.concatenate([fp_tr, t_tr], axis=1)
    X_te = np.concatenate([fp_te, t_te], axis=1)

    y_tr, _  = _targets(train)
    y_te, _  = _targets(test)

    y_scaler = StandardScaler()
    y_tr_sc  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te_sc  = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    meta = {
        "model":     "D",
        "n_bits":    n_bits,
        "radius":    radius,
        "endpoints": TOX21_SCORE_FIELDS,
        "t_scaler":  t_scaler,
        "y_scaler":  y_scaler,
        "X_shape":   X_tr.shape,
    }
    return X_tr, X_te, y_tr_sc, y_te_sc, ids_tr, ids_te, meta
