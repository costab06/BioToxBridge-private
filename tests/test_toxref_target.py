"""Unit tests for toxref_target.build_target().

Run with: pytest tests/test_toxref_target.py -v
"""

import csv
import math
import tempfile
from pathlib import Path

import pytest

from biotoxbridge.targets.toxref_target import build_target, _median


# ── helper ────────────────────────────────────────────────────────────────────

def _write_pod_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


BASE_ROW = {
    "dsstox_substance_id": "DTXSID0000001",
    "preferred_name":      "Test Chemical",
    "toxval_effect_list":  "pathology microscopic-liver-necrosis",
    "species":             "rat",
    "admin_route":         "Oral",
    "study_type":          "CHR",
    "calc_pod_type":       "LOAEL",
    "qualifier":           "",
    "mg_kg_day_value":     "10.0",
    "study_id":            "STU001",
}


# ── _median ───────────────────────────────────────────────────────────────────

def test_median_odd():
    assert _median([3.0, 1.0, 2.0]) == 2.0


def test_median_even():
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


# ── filter: effect_substring (contains, not startswith) ──────────────────────

def test_contains_filter_passes_midstring():
    """Effect substring appearing mid-string must still be accepted."""
    row = {**BASE_ROW,
           "toxval_effect_list": "urinalysis-occult blood|pathology microscopic-liver-necrosis"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        result = build_target(p)
    assert len(result) == 1


def test_startswith_only_would_fail():
    """Sanity: a row where effect only matches mid-string would fail if we used startswith."""
    row = {**BASE_ROW,
           "toxval_effect_list": "urinalysis-occult blood|pathology microscopic-liver-necrosis"}
    assert not row["toxval_effect_list"].startswith("pathology microscopic-liver-")
    assert "pathology microscopic-liver-" in row["toxval_effect_list"]


def test_non_liver_effect_excluded():
    row = {**BASE_ROW, "toxval_effect_list": "hematology-WBC-increased"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        result = build_target(p)
    assert result == []


# ── filter: species / route / study_type / pod_type / qualifier ──────────────

def test_wrong_species_excluded():
    row = {**BASE_ROW, "species": "mouse"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        assert build_target(p) == []


def test_wrong_route_excluded():
    row = {**BASE_ROW, "admin_route": "Inhalation"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        assert build_target(p) == []


def test_wrong_study_type_excluded():
    row = {**BASE_ROW, "study_type": "ACU"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        assert build_target(p) == []


def test_greater_than_qualifier_excluded():
    row = {**BASE_ROW, "qualifier": ">"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        assert build_target(p) == []


def test_non_positive_value_excluded():
    row = {**BASE_ROW, "mg_kg_day_value": "0"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        assert build_target(p) == []


def test_non_numeric_value_excluded():
    row = {**BASE_ROW, "mg_kg_day_value": "NR"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv([row], p)
        assert build_target(p) == []


# ── aggregation: CHR preference over SUB ─────────────────────────────────────

def test_chr_preferred_over_sub():
    rows = [
        {**BASE_ROW, "study_type": "CHR", "mg_kg_day_value": "5.0",
         "study_id": "CHR1"},
        {**BASE_ROW, "study_type": "SUB", "mg_kg_day_value": "20.0",
         "study_id": "SUB1"},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv(rows, p)
        result = build_target(p)
    assert len(result) == 1
    assert result[0]["selected_study_type"] == "CHR"
    assert result[0]["loael_mg_kg_day_median"] == pytest.approx(5.0)


def test_sub_used_when_no_chr():
    rows = [
        {**BASE_ROW, "study_type": "SUB", "mg_kg_day_value": "20.0",
         "study_id": "SUB1"},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv(rows, p)
        result = build_target(p)
    assert len(result) == 1
    assert result[0]["selected_study_type"] == "SUB"


# ── aggregation: median and log10 ─────────────────────────────────────────────

def test_median_aggregation():
    rows = [
        {**BASE_ROW, "mg_kg_day_value": "4.0",  "study_id": "S1"},
        {**BASE_ROW, "mg_kg_day_value": "8.0",  "study_id": "S2"},
        {**BASE_ROW, "mg_kg_day_value": "12.0", "study_id": "S3"},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv(rows, p)
        result = build_target(p)
    assert result[0]["loael_mg_kg_day_median"] == pytest.approx(8.0)
    assert result[0]["log10_loael_median"] == pytest.approx(math.log10(8.0))


def test_multi_chemical_aggregation():
    rows = [
        {**BASE_ROW, "dsstox_substance_id": "DTXSID0000001",
         "mg_kg_day_value": "10.0", "study_id": "S1"},
        {**BASE_ROW, "dsstox_substance_id": "DTXSID0000002",
         "preferred_name": "Chemical 2",
         "mg_kg_day_value": "100.0", "study_id": "S2"},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pod.csv"
        _write_pod_csv(rows, p)
        result = build_target(p)
    assert len(result) == 2
    dtxs = {r["dtxsid"] for r in result}
    assert dtxs == {"DTXSID0000001", "DTXSID0000002"}
