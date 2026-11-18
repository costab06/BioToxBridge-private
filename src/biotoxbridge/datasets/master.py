"""Assemble the liver_master.csv dataset.

Joins the ToxRefDB target table, structure-enriched resolver output, and
Tox21 predicted scores into a single per-chemical master CSV.

Every exclusion and join failure is written to an audit log so there are no
silent drops.

Usage:
    from biotoxbridge.datasets.master import build_master

    master, audit = build_master(
        target_records=target,          # from toxref_target.build_target()
        scored_records=scored,          # from tox21_adapter.score_records()
        output_csv="data/processed/liver_master.csv",
        audit_csv="data/interim/master_audit.csv",
    )
"""

import csv
import math
import sys
from pathlib import Path
from typing import Any

from biotoxbridge.features.tox21_adapter import TOX21_SCORE_FIELDS  # noqa: re-export

# Columns that constitute the final master table (in order).
MASTER_FIELDS = [
    # Identity
    "dtxsid", "preferred_name", "casrn",
    "canonical_smiles", "inchikey",
    "structure_group",
    # Target
    "selected_study_type", "n_selected_studies", "n_selected_rows",
    "loael_mg_kg_day_median", "loael_mg_kg_day_min", "loael_mg_kg_day_max",
    "log10_loael_median",
    # Tox21 scores (12 endpoints — appended dynamically)
]

ALL_MASTER_FIELDS = MASTER_FIELDS + TOX21_SCORE_FIELDS


def build_master(
    target_records: list[dict],
    scored_records: list[dict],
    output_csv: str | Path | None = None,
    audit_csv: str | Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Join target and scored structure records into the master dataset.

    Parameters
    ----------
    target_records  : output of toxref_target.build_target()
    scored_records  : output of tox21_adapter.score_records()
                      (already contains canonical_smiles, structure_group,
                       tox21_* columns, plus resolver identity fields)
    output_csv      : path to write liver_master.csv (optional)
    audit_csv       : path to write per-row audit of exclusions (optional)

    Returns
    -------
    (master, excluded)
      master   : list of dicts with ALL_MASTER_FIELDS
      excluded : list of dicts explaining each dropped chemical
    """
    # Index scored records by dtxsid
    scored_by_dtx: dict[str, dict] = {}
    for record in scored_records:
        dtxsid = record["dtxsid"]
        if dtxsid in scored_by_dtx:
            raise ValueError(f"Duplicate scored DTXSID: {dtxsid}")
        scored_by_dtx[dtxsid] = record
    target_ids = [r["dtxsid"] for r in target_records]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Duplicate target DTXSID")

    master: list[dict] = []
    excluded: list[dict] = []

    for trec in sorted(target_records, key=lambda r: r["dtxsid"]):
        dtx = trec["dtxsid"]

        if dtx not in scored_by_dtx:
            excluded.append({
                "dtxsid": dtx,
                "preferred_name": trec.get("preferred_name", ""),
                "exclusion_reason": "no_structure_or_tox21_scores",
            })
            continue

        srec = scored_by_dtx[dtx]

        # Validate every endpoint before admitting a chemical to the master.
        # Raw continuous model scores may legitimately be negative or > 1.
        missing_scores = [f for f in TOX21_SCORE_FIELDS
                          if f not in srec or srec[f] is None
                          or (isinstance(srec[f], str) and not srec[f].strip())]
        if missing_scores:
            excluded.append({
                "dtxsid": dtx,
                "preferred_name": trec.get("preferred_name", ""),
                "exclusion_reason": "missing_tox21_scores",
                "missing_fields": ",".join(missing_scores),
            })
            continue

        validated_scores = {}
        invalid_scores = []
        for field in TOX21_SCORE_FIELDS:
            try:
                value = float(srec[field])
            except (TypeError, ValueError, OverflowError):
                invalid_scores.append(field)
                continue
            if not math.isfinite(value):
                invalid_scores.append(field)
                continue
            validated_scores[field] = value
        if invalid_scores:
            excluded.append({
                "dtxsid": dtx,
                "preferred_name": trec.get("preferred_name", ""),
                "exclusion_reason": "invalid_tox21_scores",
                "invalid_fields": ",".join(invalid_scores),
            })
            continue

        row: dict[str, Any] = {}
        # Identity fields (from scored/resolved record, prefer over target)
        for f in ("dtxsid", "casrn", "inchikey"):
            row[f] = srec.get(f) or trec.get(f, "")
        row["preferred_name"]  = trec.get("preferred_name", "")
        row["canonical_smiles"] = srec.get("canonical_smiles", "")
        row["structure_group"]  = srec.get("structure_group", "")

        # Target fields
        for f in ("selected_study_type", "n_selected_studies", "n_selected_rows",
                  "loael_mg_kg_day_median", "loael_mg_kg_day_min",
                  "loael_mg_kg_day_max", "log10_loael_median"):
            row[f] = trec.get(f, "")

        # Tox21 scores
        for f in TOX21_SCORE_FIELDS:
            row[f] = validated_scores[f]

        master.append(row)

    print(
        f"Master dataset: {len(master)} chemicals included, "
        f"{len(excluded)} excluded.",
        file=sys.stderr,
    )

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ALL_MASTER_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(master)
        print(f"Master CSV written: {output_csv}", file=sys.stderr)

    if audit_csv and excluded:
        audit_csv = Path(audit_csv)
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        all_keys = list(dict.fromkeys(k for r in excluded for k in r))
        with open(audit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(excluded)
        print(f"Exclusion audit written: {audit_csv}", file=sys.stderr)

    return master, excluded
