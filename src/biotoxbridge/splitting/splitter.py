"""Group-based train/test split.

Uses sklearn GroupShuffleSplit with structure_group as the grouping key so
that no structure group spans both train and test partitions.

The split manifest is written to a JSON file for reproducibility — subsequent
pipeline runs load the frozen manifest rather than re-splitting.

Usage:
    from biotoxbridge.splitting.splitter import make_split, load_split

    manifest = make_split(master, splits_dir="data/splits", seed=42)
    train, test = load_split(master, "data/splits/split_manifest.json")
"""

import json
import sys
from pathlib import Path


def make_split(
    master: list[dict],
    splits_dir: str | Path = "data/splits",
    test_fraction: float = 0.20,
    seed: int = 42,
    group_field: str = "structure_group",
) -> dict:
    """
    Perform a single GroupShuffleSplit and persist the manifest.

    Parameters
    ----------
    master        : list of dicts from datasets.master.build_master()
    splits_dir    : directory where split_manifest.json is written
    test_fraction : fraction of *groups* assigned to test (default 0.20)
    seed          : random_state for reproducibility (default 42)
    group_field   : column holding the structure group label

    Returns
    -------
    manifest dict:
        {
          "seed": 42,
          "test_fraction": 0.20,
          "group_field": "structure_group",
          "n_train": <int>,
          "n_test":  <int>,
          "n_train_groups": <int>,
          "n_test_groups":  <int>,
          "train_dtxsids":  [...],
          "test_dtxsids":   [...],
        }
    """
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np

    dtxsids = [r["dtxsid"]      for r in master]
    groups  = [r[group_field]   for r in master]

    if len(set(dtxsids)) != len(dtxsids):
        raise ValueError("Duplicate DTXSID in master dataset")
    if any(g is None or g == "" for g in groups):
        raise ValueError("Missing structure_group in master dataset")

    X_dummy = np.zeros(len(dtxsids))  # GroupShuffleSplit only needs shape
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                            random_state=seed)
    train_idx, test_idx = next(gss.split(X_dummy, groups=groups))

    train_dtxsids = [dtxsids[i] for i in sorted(train_idx)]
    test_dtxsids  = [dtxsids[i] for i in sorted(test_idx)]
    train_groups  = {groups[i] for i in train_idx}
    test_groups   = {groups[i] for i in test_idx}

    manifest = {
        "seed":            seed,
        "test_fraction":   test_fraction,
        "group_field":     group_field,
        "n_train":         len(train_dtxsids),
        "n_test":          len(test_dtxsids),
        "n_train_groups":  len(train_groups),
        "n_test_groups":   len(test_groups),
        "train_dtxsids":   train_dtxsids,
        "test_dtxsids":    test_dtxsids,
    }

    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = splits_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"Split manifest written: {manifest_path}\n"
        f"  train: {len(train_dtxsids)} chemicals / {len(train_groups)} groups\n"
        f"  test:  {len(test_dtxsids)} chemicals / {len(test_groups)} groups",
        file=sys.stderr,
    )
    return manifest


def load_split(
    master: list[dict],
    manifest_path: str | Path,
) -> tuple[list[dict], list[dict]]:
    """
    Re-materialise train/test lists from a frozen split manifest.

    Returns (train_records, test_records) in dtxsid sort order.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    train_set = set(manifest["train_dtxsids"])
    test_set  = set(manifest["test_dtxsids"])

    ids = [r["dtxsid"] for r in master]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate DTXSID in master dataset")
    if len(train_set) != len(manifest["train_dtxsids"]) or len(test_set) != len(manifest["test_dtxsids"]):
        raise ValueError("Duplicate chemical ID in frozen split")
    if train_set & test_set:
        raise ValueError("Train/test chemical ID overlap in frozen split")
    if set(ids) != train_set | test_set:
        raise ValueError("Frozen split does not exactly match current master chemicals")
    group_field = manifest["group_field"]
    groups_train = {r[group_field] for r in master if r["dtxsid"] in train_set}
    groups_test = {r[group_field] for r in master if r["dtxsid"] in test_set}
    if None in groups_train | groups_test or "" in groups_train | groups_test:
        raise ValueError("Missing structure group in frozen split input")
    if groups_train & groups_test:
        raise ValueError("Structure group leakage in frozen split")
    if (len(train_set) != manifest["n_train"] or len(test_set) != manifest["n_test"]
            or len(groups_train) != manifest["n_train_groups"]
            or len(groups_test) != manifest["n_test_groups"]):
        raise ValueError("Frozen split counts do not match manifest")

    train = sorted([r for r in master if r["dtxsid"] in train_set],
                   key=lambda r: r["dtxsid"])
    test  = sorted([r for r in master if r["dtxsid"] in test_set],
                   key=lambda r: r["dtxsid"])
    return train, test
