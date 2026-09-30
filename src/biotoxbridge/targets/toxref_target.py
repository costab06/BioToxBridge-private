"""ToxRefDB target builder.

Applies the liver-LOAEL filtering funnel to toxrefdb_3_0_pod.csv and produces
an aggregated per-chemical target table with full audit output.

Filter logic (from legacy pipeline, verified against 191-chemical reference):
  1. toxval_effect_list contains 'pathology microscopic-liver-'
  2. species == 'rat'
  3. admin_route == 'Oral'
  4. study_type in {'CHR', 'SUB'}
  5. calc_pod_type == 'LOAEL'
  6. qualifier not in config.qualifier_exclude  (default: '>')
  7. mg_kg_day_value is positive numeric

Per-chemical aggregation:
  - Prefer CHR records when any exist; otherwise use SUB.
  - Target = median of selected LOAEL values.
  - log10_loael = log10(median LOAEL mg/kg/day).
"""

import csv
import math
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _pos_num(v: str) -> bool:
    try:
        return float(v) > 0
    except (ValueError, TypeError):
        return False


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ── public API ────────────────────────────────────────────────────────────────

def build_target(
    pod_csv: str | Path,
    effect_substring: str = "pathology microscopic-liver-",
    species: str = "rat",
    route: str = "Oral",
    study_types: tuple[str, ...] = ("CHR", "SUB"),
    duration_preference: tuple[str, ...] = ("CHR", "SUB"),
    pod_type: str = "LOAEL",
    qualifier_exclude: tuple[str, ...] = (">",),
    audit_csv: str | Path | None = None,
) -> list[dict]:
    """
    Read toxrefdb_3_0_pod.csv and return a list of per-chemical target dicts.

    Each dict contains:
        dtxsid, preferred_name, selected_study_type,
        n_selected_studies, n_selected_rows,
        loael_mg_kg_day_median, loael_mg_kg_day_min, loael_mg_kg_day_max,
        log10_loael_median

    If audit_csv is given, write a row-level audit of every filter step.
    """
    pod_csv = Path(pod_csv)
    if not pod_csv.exists():
        raise FileNotFoundError(f"ToxRefDB POD file not found: {pod_csv}")

    with open(pod_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    audit_rows = []
    counters = {"total": len(rows)}

    def _annotate(row, stage, keep):
        if audit_csv:
            audit_rows.append({**row, "_audit_stage": stage, "_audit_keep": keep})

    # ── step-by-step filter with audit ────────────────────────────────────────
    passed = []
    for r in rows:
        dtxsid  = r.get("dsstox_substance_id", "")
        effect  = r.get("toxval_effect_list", "")
        sp      = r.get("species", "")
        rte     = r.get("admin_route", "")
        stype   = r.get("study_type", "")
        pod     = r.get("calc_pod_type", "")
        qual    = r.get("qualifier", "")
        val_str = r.get("mg_kg_day_value", "")

        if effect_substring not in effect:
            continue
        if sp != species:
            continue
        if rte != route:
            continue
        if stype not in study_types:
            continue
        if pod != pod_type:
            continue
        if qual in qualifier_exclude:
            continue
        if not _pos_num(val_str):
            continue
        passed.append(r)

    counters["passed_all_filters"] = len(passed)
    counters["distinct_chemicals"]  = len({r["dsstox_substance_id"] for r in passed})

    # ── per-chemical aggregation ───────────────────────────────────────────────
    from collections import defaultdict
    chem_rows: dict[str, list] = defaultdict(list)
    for r in passed:
        chem_rows[r["dsstox_substance_id"]].append(r)

    results = []
    for dtxsid, data in chem_rows.items():
        # Duration preference
        for preferred_type in duration_preference:
            selected = [r for r in data if r["study_type"] == preferred_type]
            if selected:
                break
        else:
            selected = data  # fallback: all

        loaels = [float(r["mg_kg_day_value"]) for r in selected]
        med    = _median(loaels)

        results.append({
            "dtxsid":                  dtxsid,
            "preferred_name":          selected[0]["preferred_name"],
            "selected_study_type":     selected[0]["study_type"],
            "n_selected_studies":      len({r["study_id"] for r in selected}),
            "n_selected_rows":         len(selected),
            "loael_mg_kg_day_median":  med,
            "loael_mg_kg_day_min":     min(loaels),
            "loael_mg_kg_day_max":     max(loaels),
            "log10_loael_median":      math.log10(med),
        })

    # Sort by DTXSID for deterministic output
    results.sort(key=lambda x: x["dtxsid"])

    # ── optional audit output ─────────────────────────────────────────────────
    if audit_csv:
        audit_csv = Path(audit_csv)
        audit_csv.parent.mkdir(parents=True, exist_ok=True)

        # Write filter-step counts as a summary header block + per-chemical rows
        with open(audit_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "dtxsid", "preferred_name", "selected_study_type",
                "n_selected_studies", "n_selected_rows",
                "loael_mg_kg_day_median", "log10_loael_median",
                "loael_mg_kg_day_min", "loael_mg_kg_day_max",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

        # Write filter step count summary alongside
        summary_path = audit_csv.parent / (audit_csv.stem + "_filter_summary.json")
        import json
        summary_path.write_text(json.dumps({
            "raw_file":            str(pod_csv),
            "total_input_rows":    counters["total"],
            "passed_all_filters":  counters["passed_all_filters"],
            "distinct_chemicals":  counters["distinct_chemicals"],
            "aggregated_chemicals": len(results),
            "filters": {
                "effect_substring":    effect_substring,
                "species":             species,
                "route":               route,
                "study_types":         list(study_types),
                "pod_type":            pod_type,
                "qualifier_exclude":   list(qualifier_exclude),
            },
        }, indent=2))

    return results


def compare_to_legacy(results: list[dict],
                      legacy_csv_paths: list[str | Path],
                      tol: float = 0.001) -> dict:
    """
    Compare build_target output against legacy CSV files.
    Returns a report dict with match/mismatch counts.
    """
    legacy: dict[str, float] = {}
    for path in legacy_csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                legacy[row["dtxsid"]] = float(row["loael_mg_kg_day_median"])

    new_set = {r["dtxsid"] for r in results}
    leg_set = set(legacy)

    only_new   = new_set - leg_set
    only_legacy = leg_set - new_set
    common      = new_set & leg_set

    mismatches = []
    for r in results:
        if r["dtxsid"] in legacy:
            diff = abs(r["loael_mg_kg_day_median"] - legacy[r["dtxsid"]])
            if diff > tol:
                mismatches.append({
                    "dtxsid":  r["dtxsid"],
                    "new":     r["loael_mg_kg_day_median"],
                    "legacy":  legacy[r["dtxsid"]],
                    "abs_diff": diff,
                })

    return {
        "n_new":      len(new_set),
        "n_legacy":   len(leg_set),
        "n_common":   len(common),
        "only_new":   sorted(only_new),
        "only_legacy": sorted(only_legacy),
        "n_mismatches": len(mismatches),
        "mismatches":  mismatches,
    }
