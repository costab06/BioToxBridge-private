"""Unit tests for splitting.splitter."""

import json
import tempfile
from pathlib import Path

import pytest

from biotoxbridge.splitting.splitter import make_split, load_split


def _make_master(n: int) -> list[dict]:
    """Create n fake master records with structure_group alternating in pairs."""
    return [
        {
            "dtxsid":          f"DTXSID{i:07d}",
            "log10_loael_median": float(i),
            "structure_group": i // 2,   # pairs share a group
        }
        for i in range(n)
    ]


def test_make_split_sizes():
    master = _make_master(20)
    with tempfile.TemporaryDirectory() as d:
        manifest = make_split(master, splits_dir=d, test_fraction=0.2, seed=42)
    n_total = manifest["n_train"] + manifest["n_test"]
    assert n_total == 20
    assert manifest["n_test"] >= 2


def test_make_split_no_group_leakage():
    master = _make_master(20)
    with tempfile.TemporaryDirectory() as d:
        manifest = make_split(master, splits_dir=d, test_fraction=0.2, seed=42)
    train_groups = set()
    test_groups  = set()
    by_dtx = {r["dtxsid"]: r["structure_group"] for r in master}
    for dtx in manifest["train_dtxsids"]:
        train_groups.add(by_dtx[dtx])
    for dtx in manifest["test_dtxsids"]:
        test_groups.add(by_dtx[dtx])
    assert train_groups.isdisjoint(test_groups), \
        "Structure groups leaked between train and test!"


def test_make_split_deterministic():
    master = _make_master(20)
    with tempfile.TemporaryDirectory() as d:
        m1 = make_split(master, splits_dir=d, test_fraction=0.2, seed=42)
    with tempfile.TemporaryDirectory() as d:
        m2 = make_split(master, splits_dir=d, test_fraction=0.2, seed=42)
    assert m1["train_dtxsids"] == m2["train_dtxsids"]
    assert m1["test_dtxsids"]  == m2["test_dtxsids"]


def test_load_split_covers_all():
    master = _make_master(20)
    with tempfile.TemporaryDirectory() as d:
        manifest = make_split(master, splits_dir=d, seed=42)
        train, test = load_split(master, Path(d) / "split_manifest.json")
    all_dtxsids = {r["dtxsid"] for r in train + test}
    assert all_dtxsids == {r["dtxsid"] for r in master}


def test_load_split_no_overlap():
    master = _make_master(20)
    with tempfile.TemporaryDirectory() as d:
        make_split(master, splits_dir=d, seed=42)
        train, test = load_split(master, Path(d) / "split_manifest.json")
    train_ids = {r["dtxsid"] for r in train}
    test_ids  = {r["dtxsid"] for r in test}
    assert train_ids.isdisjoint(test_ids)
