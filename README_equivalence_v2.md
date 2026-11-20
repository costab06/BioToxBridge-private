# BioToxBridge OAF equivalence verifier v2

Replaces `scripts/verify_oaf_equivalence.py` from v1. The verifier now treats empty strings and null as equivalent **only** for `inchiString`, `inchikey`, and `msReadySmiles` within `missing` records marked `failure_reason=qsarReadySmiles_missing`. The report includes `normalizations` entries; all other comparisons remain exact. The original OAF artifacts are never modified.

Run from the repository root with the BioToxBridge virtual environment active:

```bash
unzip -o ~/Downloads/biotoxbridge_oaf_equivalence_v2.zip
python -m pytest -q tests/test_verify_oaf_equivalence.py
python scripts/verify_oaf_equivalence.py \\
  --config config/experiments/liver_loael_v1.yaml \\
  --run-id oaf-test-002 \\
  --artifact-root runs/oaf-artifacts \\
  --direct-work-dir runs/equivalence-direct-003 \\
  --cache-csv runs/oaf-work/oaf-test-002/interim/dsstox_cache.csv \\
  --report runs/equivalence-report-003.json
```

This reruns Tox21 scoring and requires the inference service. Use a fresh direct-work-dir because the verifier intentionally refuses to overwrite previous results.
