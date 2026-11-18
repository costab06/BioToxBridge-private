"""Safety checks required before durable workflow integration."""
import json
import pytest
from biotoxbridge.splitting.splitter import make_split, load_split
from biotoxbridge.datasets.master import build_master


def records():
    return [{"dtxsid": f"D{i}", "structure_group": f"G{i//2}"} for i in range(20)]


def test_changed_chemical_population_rejected(tmp_path):
    data = records()
    make_split(data, splits_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly match"):
        load_split(data[:-1], tmp_path / "split_manifest.json")


def test_group_leakage_rejected(tmp_path):
    data = records()
    make_split(data, splits_dir=tmp_path)
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    train_id = manifest["train_dtxsids"][0]
    test_id = manifest["test_dtxsids"][0]
    by_id = {r["dtxsid"]: r for r in data}
    by_id[test_id]["structure_group"] = by_id[train_id]["structure_group"]
    with pytest.raises(ValueError, match="leakage"):
        load_split(data, tmp_path / "split_manifest.json")


def test_duplicate_master_chemical_rejected(tmp_path):
    data = records()
    make_split(data, splits_dir=tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        load_split(data + [data[0]], tmp_path / "split_manifest.json")


def test_duplicate_scored_records_rejected():
    with pytest.raises(ValueError, match="Duplicate scored"):
        build_master([], [{"dtxsid": "D1"}, {"dtxsid": "D1"}])
