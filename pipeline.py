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
    run_dir       = Path("runs") / exp_id
    audit_dir     = run_dir / "audit"
    for d in (interim_dir, processed_dir, splits_dir, run_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: ToxRefDB target ────────────────────────────────────────────────
    print("\n=== Step 1: ToxRefDB target ===")
    from biotoxbridge.targets.toxref_target import build_target
    tc = cfg["target"]
    target = build_target(
        pod_csv             = tc["raw_file"],
        effect_substring    = tc["effect_substring"],
        species             = tc["species"],
        route               = tc["route"],
        study_types         = tuple(tc["study_types"]),
        duration_preference = tuple(tc["duration_preference"]),
        pod_type            = tc["pod_type"],
        qualifier_exclude   = tuple(tc.get("qualifier_exclude", [">"])),
        audit_csv           = audit_dir / "toxref_target.csv",
    )
    print(f"  Target chemicals: {len(target)}")

    # ── Step 2: Chemical resolver ──────────────────────────────────────────────
    print("\n=== Step 2: Chemical resolver ===")
    from biotoxbridge.chemistry.resolver import resolve, write_audit
    dtxsids = [r["dtxsid"] for r in target]
    resolved, missing = resolve(
        dtxsids,
        cache_csv = interim_dir / "dsstox_cache.csv",
    )
    write_audit(
        resolved, missing,
        resolved_path = audit_dir / "resolver_resolved.csv",
        missing_path  = audit_dir / "resolver_missing.csv",
    )
    print(f"  Resolved: {len(resolved)}  Missing: {len(missing)}")

    # ── Step 3: Structure groups + fingerprints ────────────────────────────────
    print("\n=== Step 3: Structure groups ===")
    from biotoxbridge.chemistry.structure_groups import assign_groups, group_summary
    enriched, sg_failed = assign_groups(
        resolved, smiles_field=cfg["chemistry"]["structure_field"]
    )
    print(f"  {group_summary(enriched)}")

    # ── Step 4: Tox21 scoring ──────────────────────────────────────────────────
    print("\n=== Step 4: Tox21 scoring ===")
    from biotoxbridge.features.tox21_adapter import Tox21Adapter
    tp = cfg["tox21_project"]
    adapter = Tox21Adapter(
        base_url  = tp["nnsd_client_url"],
        endpoints = tp["endpoints"],
    )
    scored = adapter.score_records(enriched, smiles_field="canonical_smiles")

    # Regression-test against legacy predictions if available
    ref_csv = Path("data/reference/legacy_pipeline/tox21_predictions.csv")
    if ref_csv.exists():
        report = adapter.regression_test(scored, str(ref_csv), tol=0.01)
        (audit_dir / "tox21_regression.json").write_text(
            json.dumps(report, indent=2)
        )
        status = "PASS" if report["all_pass"] else "FAIL"
        print(f"  Tox21 regression test: {status}")

    # ── Step 5: Master dataset ─────────────────────────────────────────────────
    print("\n=== Step 5: Master dataset ===")
    from biotoxbridge.datasets.master import build_master
    master, excluded = build_master(
        target_records = target,
        scored_records = scored,
        output_csv     = processed_dir / "liver_master.csv",
        audit_csv      = audit_dir / "master_exclusions.csv",
    )
    print(f"  Master: {len(master)} chemicals")

    if not master:
        print("ERROR: empty master dataset — check audit logs.", file=sys.stderr)
        sys.exit(1)

    # ── Step 6: Split ──────────────────────────────────────────────────────────
    print("\n=== Step 6: Train/test split ===")
    from biotoxbridge.splitting.splitter import make_split, load_split
    manifest_path = splits_dir / "split_manifest.json"

    if reuse_split and manifest_path.exists():
        print(f"  Loading frozen split from {manifest_path}")
        train, test = load_split(master, manifest_path)
    else:
        sp = cfg["split"]
        manifest = make_split(
            master,
            splits_dir    = splits_dir,
            test_fraction = sp["test_fraction"],
            seed          = sp["seed"],
            group_field   = sp["group_by"],
        )
        train, test = load_split(master, manifest_path)

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
        source_manifest   = src_manifest,
        artifact_paths    = {
            "liver_master": processed_dir / "liver_master.csv",
            "predictions":  pred_path,
            "metrics":      metrics_path,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioToxBridge liver LOAEL pipeline")
    parser.add_argument("config", help="Path to experiment YAML")
    parser.add_argument("--reuse-split", action="store_true",
                        help="Load existing split manifest instead of re-splitting")
    args = parser.parse_args()
    run(args.config, reuse_split=args.reuse_split)
