# BioToxBridge OAF Phase 2 — training, evaluation, reporting

Adds `model_training`, `model_evaluation`, `experiment_reporting` to the six validated preparation stages, using existing BioToxBridge projectors, RidgeCV trainer and evaluator. No OpenAgentFabric changes.

## Install

From repository root:

```bash
unzip -o ~/Downloads/biotoxbridge_oaf_modeling_phase2_v1.zip
python scripts/apply_oaf_modeling_phase2.py
python -m pytest -q tests/test_oaf_modeling.py
python -m pytest -q
```

**Important:** Existing coordinator tests that assume `dataset_split` is terminal must be updated to expect `model_training` next and `experiment_reporting` as terminal. Do not weaken validation tests. The full test suite is not expected to pass unchanged if those old expectations remain.

Run the OAF runtime with a **new run ID**; do not resume an old six-stage run. Tox21 inference and Redis 7 must be running. The final OAF completion event now follows reporting, and contains the train/test counts from the report.

## Scientific behavior

- Models A/B/D use `datasets.projectors` with training-only scaler fits and the frozen split.
- `training.trainer.train_model` performs RidgeCV on training records.
- `evaluation.evaluator.evaluate` and `bootstrap_ci` evaluate original-scale test predictions.
- Model predictions and metrics are handed off through OAF JSON artifacts. Final files `metrics.json` and `predictions.csv` are written under the run work directory.
- Non-finite statistics become JSON `null`, preserving valid immutable JSON. Model estimators are not serialized by this first increment; model training on crash before output publication may rerun.

## Limitations / next verification

This patch has not been run against the user's local BioToxBridge, Redis, or Tox21 service. Check stage-by-stage scientific equivalence with the standalone `pipeline.py` after a complete run; training and evaluation have not yet been validated as equivalent. Existing crash test harness assumes `dataset_split` is terminal and must be adapted for the new final stage. The reporting CSV files are convenience outputs; the immutable OAF JSON result is the authoritative final artifact.
