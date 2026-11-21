"""Immutable, checksum-verified JSON handoffs for OAF workflows.

OAF's current ArtifactType enum has no scientific-data type; use its existing
EVALUATION value for JSON stage handoffs until an explicit contract is agreed.
This module intentionally does not depend on OAF, allowing standalone tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _encoded(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _safe_path(root: Path, run_id: str, stage: str) -> Path:
    for value in (run_id, stage):
        if not value or value in (".", "..") or not all(
            c.isascii() and (c.isalnum() or c in "-_.") for c in value
        ):
            raise ValueError("run_id and stage must be safe nonempty path components")
    return root / run_id / f"{stage}.json"


def publish_json(root: str | Path, run_id: str, stage: str, data: Any) -> dict[str, str]:
    """Create a durable stage artifact without overwriting existing results.

    Returns an OAF-compatible reference payload; OAF ArtifactRef validation is
    performed by the OAF adapter, not by this storage contract.
    """
    root = Path(root).resolve()
    path = _safe_path(root, run_id, stage)
    payload = _encoded(data)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Artifact conflict: {path}")
    return {"artifact_id": f"{run_id}/{stage}.json", "artifact_type": "evaluation",
            "uri": str(path), "version": 1, "checksum": digest}


def read_json(reference: dict, root: str | Path) -> Any:
    """Verify artifact containment, identity and SHA-256 before parsing."""
    root = Path(root).resolve()
    artifact_id = reference["artifact_id"]
    parts = artifact_id.split("/")
    if len(parts) != 2 or not parts[1].endswith(".json"):
        raise ValueError("Invalid artifact identifier")
    expected = _safe_path(root, parts[0], parts[1][:-5]).resolve()
    actual = Path(reference["uri"]).resolve()
    if actual != expected or not actual.is_relative_to(root):
        raise ValueError("Artifact path does not match its identity or root")
    payload = actual.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference["checksum"]:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(payload)
