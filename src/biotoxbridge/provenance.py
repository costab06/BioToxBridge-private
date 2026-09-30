"""Source registration, SHA-256 checksums, and run manifests."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_source(path: str | Path, source: str, release: str,
                    retrieval_date: str, description: str = "") -> dict:
    """Return a provenance record for an immutable source file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")
    return {
        "file":           str(p),
        "sha256":         sha256(p),
        "size_bytes":     p.stat().st_size,
        "source":         source,
        "release":        release,
        "retrieval_date": retrieval_date,
        "description":    description,
        "registered_at":  datetime.now(timezone.utc).isoformat(),
    }


def load_source_manifest(path: str | Path) -> dict:
    """Load an existing source manifest; return empty dict if not found."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_source_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2))


def _git_commit(repo_dir: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_run_manifest(run_dir: str | Path, config_path: str | Path,
                       source_manifest: dict, artifact_paths: dict,
                       extra: dict | None = None) -> None:
    """Write run_manifest.json recording config/data/model hashes and versions."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {k: {"path": str(v), "sha256": sha256(v)}
                 for k, v in artifact_paths.items() if Path(v).exists()}

    manifest = {
        "git_commit":       _git_commit(),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "python_version":   sys.version,
        "platform":         platform.platform(),
        "config_file":      str(config_path),
        "config_sha256":    sha256(config_path),
        "source_manifest":  source_manifest,
        "artifacts":        artifacts,
    }
    if extra:
        manifest.update(extra)

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Run manifest written: {run_dir / 'run_manifest.json'}")
