"""EPA CTX Chemical API resolver.

Fetches QSAR-ready SMILES and DSSTox identity fields for a list of DTXSIDs.
Results are cached to a local CSV to avoid redundant API calls.

API: POST https://comptox.epa.gov/ctx-api/chemical/detail/search/by-dtxsid/
     ?projection=chemicalstructure
Header: x-api-key: <EPA_CTX_API_KEY>
Body: ["DTXSID...", ...]
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

FIELDS = [
    "dtxsid", "dtxcid", "casrn", "preferredName",
    "smiles", "qsarReadySmiles", "msReadySmiles",
    "inchikey", "inchiString",
]

_API_BASE  = "https://comptox.epa.gov/ctx-api"
_API_PATH  = "/chemical/detail/search/by-dtxsid/"
_BATCH_SZ  = 50
_TIMEOUT   = 60
_RETRIES   = 4


def _batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _request_batch(session, url: str, key: str,
                   ids: list[str], retries: int) -> list[dict]:
    import requests
    headers = {
        "x-api-key":     key,
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    for attempt in range(retries + 1):
        try:
            r = session.post(
                url,
                params={"projection": "chemicalstructure"},
                headers=headers,
                json=ids,
                timeout=_TIMEOUT,
            )
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in ("data", "results", "content"):
                    if isinstance(data.get(k), list):
                        return data[k]
                if "dtxsid" in data:
                    return [data]
            raise RuntimeError("Unexpected response shape: " + json.dumps(data)[:300])
        except Exception as e:
            if attempt == retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"  retry ({attempt + 1}/{retries}) after: {e} ({wait}s)", file=sys.stderr)
            time.sleep(wait)
    return []  # unreachable


def resolve(
    dtxsids: list[str],
    cache_csv: str | Path | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Resolve DTXSIDs to structure records.

    Returns (resolved, missing) where:
      resolved: list of dicts with FIELDS; qsarReadySmiles is non-empty
      missing:  list of dicts with dtxsid and failure_reason

    If cache_csv exists and covers all requested DTXSIDs, skips the API call.
    If cache_csv is given but incomplete, fetches only the missing subset and
    appends to the cache.
    """
    import requests

    dtxsids = list(dict.fromkeys(d.strip() for d in dtxsids if d.strip()))

    # ── check cache ────────────────────────────────────────────────────────────
    cached: dict[str, dict] = {}
    if cache_csv and Path(cache_csv).exists():
        with open(cache_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cached[row.get("dtxsid", row.get("_epa_dtxsid", ""))] = row

    to_fetch = [d for d in dtxsids if d not in cached]

    if to_fetch:
        key = api_key or os.environ.get("EPA_CTX_API_KEY")
        if not key:
            raise RuntimeError(
                "EPA_CTX_API_KEY environment variable not set and no api_key provided. "
                "Set the variable or supply a pre-built cache_csv."
            )

        url     = _API_BASE.rstrip("/") + _API_PATH
        session = requests.Session()
        bs      = list(_batches(to_fetch, _BATCH_SZ))
        fetched = []
        failed  = []

        print(f"Fetching {len(to_fetch)} DTXSIDs from EPA CTX API ({len(bs)} batches)...")
        for i, batch in enumerate(bs, 1):
            print(f"  Batch {i}/{len(bs)} ({len(batch)} IDs)")
            try:
                recs = _request_batch(session, url, key, batch, _RETRIES)
                fetched.extend(recs)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                failed.extend(batch)

        # Normalise returned records to the FIELDS schema
        new_records = []
        for rec in fetched:
            normed = {f: rec.get(f, "") for f in FIELDS}
            # API may return dtxsid in different capitalisation
            for k in ("dtxsid", "DTXSID", "dsstox_substance_id"):
                if rec.get(k):
                    normed["dtxsid"] = str(rec[k]).strip()
                    break
            qs = str(normed.get("qsarReadySmiles") or "").strip()
            normed["qsarReadySmiles"] = qs
            new_records.append(normed)

        # Update cache and write
        for rec in new_records:
            cached[rec["dtxsid"]] = rec

        if cache_csv:
            cache_csv = Path(cache_csv)
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(cached.values())
            print(f"Cache written: {cache_csv} ({len(cached)} records)")

    # ── build resolved / missing ───────────────────────────────────────────────
    resolved = []
    missing  = []

    for dtx in dtxsids:
        rec = cached.get(dtx)
        if rec is None:
            missing.append({"dtxsid": dtx, "failure_reason": "not_in_cache_or_api"})
            continue
        qs = str(rec.get("qsarReadySmiles") or "").strip()
        if not qs:
            missing.append({**rec, "failure_reason": "qsarReadySmiles_missing"})
        else:
            resolved.append({f: rec.get(f, "") for f in FIELDS})

    return resolved, missing


def write_audit(resolved: list[dict], missing: list[dict],
                resolved_path: str | Path, missing_path: str | Path) -> None:
    """Write resolver outputs to CSVs for audit."""
    for path, rows, fields in [
        (resolved_path, resolved, FIELDS),
        (missing_path,  missing,  ["dtxsid", "failure_reason"] + FIELDS[1:]),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Gather all keys present
        all_keys = list(dict.fromkeys(
            k for r in rows for k in r
        )) if rows else fields
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
