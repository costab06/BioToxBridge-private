"""Tox21Project model adapter.

Scores chemical structures against the 12 continuous Tox21 models by sending
Morgan fingerprints to the nnsd_client HTTP server.

API contract:
  POST {base_url}/api/models/{model_name}/predict
  Body: {"features": [<2048 floats>], "categorical": false}
  Response: {"value": <float>}   (raw continuous score, already de-scaled)

Usage:
    from biotoxbridge.features.tox21_adapter import Tox21Adapter

    adapter = Tox21Adapter("http://localhost:8080")
    scores  = adapter.score_records(enriched_records)   # list[dict]
"""

import sys
import time
from typing import Any

# Canonical endpoint order — must match the frozen model set exactly.
TOX21_ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

_RETRIES = 3
_TIMEOUT = 30


class Tox21Adapter:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        endpoints: list[str] | None = None,
        radius: int = 2,
        n_bits: int = 2048,
        retries: int = _RETRIES,
        timeout: int = _TIMEOUT,
    ):
        self.base_url  = base_url.rstrip("/")
        self.endpoints = endpoints or TOX21_ENDPOINTS
        self.radius    = radius
        self.n_bits    = n_bits
        self.retries   = retries
        self.timeout   = timeout

    # ── low-level ─────────────────────────────────────────────────────────────

    def _predict_one(self, session: Any, model_name: str,
                     features: list[float]) -> float:
        import requests
        url = f"{self.base_url}/api/models/{model_name}/predict"
        payload = {"features": features, "categorical": False}
        for attempt in range(self.retries + 1):
            try:
                r = session.post(url, json=payload, timeout=self.timeout)
                if not r.ok:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                if "value" in data:
                    return float(data["value"])
                raise RuntimeError(f"Unexpected response: {data}")
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = min(2 ** attempt, 10)
                print(f"  retry ({attempt+1}/{self.retries}) {model_name}: {e} ({wait}s)",
                      file=sys.stderr)
                time.sleep(wait)
        return float("nan")   # unreachable

    # ── public API ────────────────────────────────────────────────────────────

    def score_smiles(self, smiles: str) -> dict[str, float]:
        """Score a single SMILES string. Returns {endpoint: value, ...}."""
        import requests
        from biotoxbridge.features.fingerprints import morgan_fp
        features = morgan_fp(smiles, radius=self.radius, n_bits=self.n_bits).tolist()
        session  = requests.Session()
        return {ep: self._predict_one(session, ep, features)
                for ep in self.endpoints}

    def score_records(
        self,
        records: list[dict],
        smiles_field: str = "canonical_smiles",
        progress: bool = True,
    ) -> list[dict]:
        """
        Score each record against all 12 endpoints.

        Returns a list of dicts with original record fields plus one column per
        endpoint named ``tox21_<endpoint>`` (e.g. ``tox21_NR-AR``).

        Records where fingerprint generation fails are skipped and logged.
        """
        import requests
        from biotoxbridge.features.fingerprints import morgan_fp

        session  = requests.Session()
        results  = []
        n        = len(records)

        for i, rec in enumerate(records, 1):
            dtx    = rec.get("dtxsid", f"row{i}")
            smiles = str(rec.get(smiles_field) or "").strip()
            if not smiles:
                print(f"  SKIP {dtx}: empty {smiles_field}", file=sys.stderr)
                continue
            try:
                features = morgan_fp(
                    smiles, radius=self.radius, n_bits=self.n_bits
                ).tolist()
            except Exception as e:
                print(f"  SKIP {dtx}: fingerprint failed: {e}", file=sys.stderr)
                continue

            scores: dict[str, float] = {}
            for ep in self.endpoints:
                scores[f"tox21_{ep}"] = self._predict_one(session, ep, features)

            results.append({**rec, **scores})
            if progress and i % 10 == 0:
                print(f"  scored {i}/{n}", file=sys.stderr)

        if progress:
            print(f"Tox21 scoring complete: {len(results)}/{n} records scored.",
                  file=sys.stderr)
        return results

    # ── regression-test ───────────────────────────────────────────────────────

    def regression_test(
        self,
        scored: list[dict],
        reference_csv: str,
        tol: float = 0.01,
    ) -> dict:
        """
        Compare scored output against a reference CSV (tox21_predictions.csv).

        Returns a report dict with per-endpoint max_abs_diff and pass/fail.
        """
        import csv
        import math

        ref: dict[str, dict] = {}
        with open(reference_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref[row["dtxsid"]] = row

        report: dict[str, Any] = {"tol": tol, "endpoints": {}}
        all_pass = True

        for ep in self.endpoints:
            col = f"tox21_{ep}"
            diffs = []
            for rec in scored:
                dtx = rec.get("dtxsid", "")
                if dtx not in ref:
                    continue
                ref_val = ref[dtx].get(col)
                if ref_val is None or ref_val == "":
                    continue
                diff = abs(rec[col] - float(ref_val))
                diffs.append(diff)
            max_diff = max(diffs) if diffs else float("nan")
            passed   = (not math.isnan(max_diff)) and (max_diff <= tol)
            if not passed:
                all_pass = False
            report["endpoints"][ep] = {
                "n_compared": len(diffs),
                "max_abs_diff": max_diff,
                "pass": passed,
            }

        report["all_pass"] = all_pass
        return report
