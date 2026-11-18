"""BioToxBridge pipeline — liver LOAEL experiment.

Runs the full sequence:
  1. ToxRefDB target construction
  2. EPA CTX chemical resolution (with cache)
  3. Structure canonicalization + group assignment
  4. Tox21 scoring
  5. Master dataset assembly
  6. Group-based train/test split (or load frozen manifest)
  7. Model A / B / D projection
  8. Train + evaluate; write metrics and predictions

Usage:
    python pipeline.py config/experiments/liver_loael_v1.yaml [--reuse-split]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: str, reuse_split: bool = False) -> None:
    cfg = _load_config(config_path)
    exp_id = cfg["experiment_id"]

    # ── directory layout ───────────────────────────────────────────────────────
    interim_dir   = Path(cfg["output"]["interim_dir"])
    processed_dir = Path(cfg["output"]["processed_dir"])
    splits_dir    = Path(cfg["output"]["splits_dir"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    run_dir       = Path("runs") / exp_id / run_id
    audit_dir     = run_dir / "audit"
    # Keep experiment outputs isolated: no run writes a shared master/split path.
    processed_dir = run_dir / "processed"
    splits_dir = run_dir / "splits"
    if reuse_split:
        raise ValueError("--reuse-split requires a future explicit, validated split source; refusing implicit reuse")
    for d in (interim_dir, processed_dir, splits_dir, run_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Scientific stages are independently callable; orchestration stays here.
    from biotoxbridge.stages import (
        build_toxicity_target, resolve_chemistry, group_structures,
        score_tox21, assemble_master, freeze_split,
    )
    from biotoxbridge.chemistry.structure_groups import group_summary

    print("\n=== Step 1: ToxRefDB target ===")
    target = build_toxicity_target(cfg, audit_dir)
    print(f"  Target chemicals: {len(target)}")

    print("\n=== Step 2: Chemical resolver ===")
    resolved, missing = resolve_chemistry(target, interim_dir, audit_dir)
    print(f"  Resolved: {len(resolved)}  Missing: {len(missing)}")

    print("\n=== Step 3: Structure groups ===")
    enriched, sg_failed = group_structures(resolved, cfg)
    print(f"  {group_summary(enriched)}")

    print("\n=== Step 4: Tox21 scoring ===")
    scored = score_tox21(enriched, cfg, audit_dir)

    print("\n=== Step 5: Master dataset ===")
    master, excluded = assemble_master(target, scored, processed_dir, audit_dir)
    print(f"  Master: {len(master)} chemicals")
    if not master:
        raise ValueError("Empty master dataset — check audit logs")

    print("\n=== Step 6: Train/test split ===")
    train, test = freeze_split(master, cfg, splits_dir)
    print(f"  Train: {len(train)}  Test: {len(test)}")

    # ── Steps 7 + 8: Project → Train → Evaluate ───────────────────────────────
    from biotoxbridge.datasets.projectors import project_A, project_B, project_D
    from biotoxbridge.training.trainer    import train_model
    from biotoxbridge.evaluation.evaluator import evaluate, bootstrap_ci, compare_models

    projectors = {
        "A": lambda: project_A(train, test),
        "B": lambda: project_B(train, test),
        "D": lambda: project_D(train, test),
    }

    all_metrics: dict[str, dict] = {}
    predictions_rows = []

    for model_id in cfg.get("models", ["structure_only",
                                        "predicted_bioactivity",
                                        "structure_plus_predicted"]):
        # map friendly names to A/B/D
        mid = {"structure_only": "A",
               "predicted_bioactivity": "B",
               "structure_plus_predicted": "D"}.get(model_id, model_id)

        if mid not in projectors:
            print(f"  Unknown model {mid}, skipping.", file=sys.stderr)
            continue

        print(f"\n=== Model {mid} ===")
        X_tr, X_te, y_tr, y_te, ids_tr, ids_te, proj_meta = projectors[mid]()

        pipeline, train_info = train_model(X_tr, y_tr, model_id=mid)
        y_pred = pipeline.predict(X_te)

        y_sc = proj_meta["y_scaler"]
        metrics = evaluate(y_te, y_pred, label=f"Model {mid}", y_scaler=y_sc)
        ci      = bootstrap_ci(
            y_te, y_pred,
            n_iter = cfg["evaluation"]["bootstrap_iterations"],
            seed   = cfg["evaluation"]["bootstrap_seed"],
            y_scaler = y_sc,
        )
        all_metrics[mid] = {**metrics, "ci": ci}

        print(f"  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
              f"R²={metrics['r2']:.3f}  ρ={metrics['spearman_rho']:.3f}")

        # Collect per-compound predictions (original scale)
        y_te_orig   = y_sc.inverse_transform(y_te.reshape(-1, 1)).ravel()
        y_pred_orig = y_sc.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        for dtx, yt, yp in zip(ids_te, y_te_orig, y_pred_orig):
            predictions_rows.append({
                "dtxsid": dtx,
                "model":  mid,
                "y_true": yt,
                "y_pred": yp,
                "residual": float(yt - yp),
            })

    # ── Write outputs ──────────────────────────────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics written: {metrics_path}")

    pred_path = run_dir / "predictions.csv"
    if predictions_rows:
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dtxsid", "model",
                                               "y_true", "y_pred", "residual"])
            w.writeheader()
            w.writerows(predictions_rows)
        print(f"Predictions written: {pred_path}")

    # ── Comparison table ───────────────────────────────────────────────────────
    plain = {k: {kk: vv for kk, vv in v.items() if kk != "ci"}
             for k, v in all_metrics.items()}
    table = compare_models(plain)
    print("\n=== Model comparison ===")
    print(f"{'Metric':<14}", end="")
    for mid in sorted(all_metrics):
        print(f"  Model {mid:>1}", end="")
    print()
    for metric, vals in table.items():
        print(f"{metric:<14}", end="")
        for mid in sorted(all_metrics):
            print(f"  {vals.get(mid, float('nan')):>8.4f}", end="")
        print()

    # ── Run manifest ───────────────────────────────────────────────────────────
    from biotoxbridge.provenance import write_run_manifest
    from biotoxbridge.provenance import load_source_manifest
    src_manifest = load_source_manifest(
        Path("data/interim") / "source_manifest.json"
    )
    write_run_manifest(
        run_dir,
        config_path       = config_path,
        extra             = {"run_id": run_id},
        source_manifest   = src_manifest,
        input_paths       = {
            "toxrefdb_pod": cfg["target"]["raw_file"],
            "tox21_reference": Path("data/reference/legacy_pipeline/tox21_predictions.csv"),
        },
        model_metadata    = {
            "model_set_id": cfg["tox21_project"]["model_set_id"],
            "endpoints": cfg["tox21_project"]["endpoints"],
            "fingerprint": cfg["chemistry"]["fingerprint"],
            "split": cfg["split"],
        },
        artifact_paths    = {
            "liver_master": processed_dir / "liver_master.csv",
            "split_manifest": splits_dir / "split_manifest.json",
            "tox21_regression_audit": audit_dir / "tox21_regression.json",
            "metrics": metrics_path,
            **({"predictions": pred_path} if predictions_rows else {}),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioToxBridge liver LOAEL pipeline")
    parser.add_argument("config", help="Path to experiment YAML")
    parser.add_argument("--reuse-split", action="store_true",
                        help="Load existing split manifest instead of re-splitting")
    args = parser.parse_args()
    run(args.config, reuse_split=args.reuse_split)
