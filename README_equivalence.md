# BioToxBridge OAF scientific equivalence verification

Run from the repository root. Requires the same Python environment and running Tox21 inference service used for the OAF run. This verifier invokes the six scientific preparation stages directly, without Redis or OAF agents, then compares each stage's data to the checksum-verified OAF artifact. Record-list ordering is ignored, but numerical arrays and field values are compared exactly.

```bash
python scripts/verify_oaf_equivalence.py \
  --config config/experiments/liver_loael_v1.yaml \
  --run-id oaf-test-002 \
  --artifact-root runs/oaf-artifacts \
  --direct-work-dir runs/equivalence-direct-002 \
  --cache-csv runs/oaf-work/oaf-test-002/interim/dsstox_cache.csv \
  --report runs/equivalence-report-002.json
```

A fresh direct output directory is required, to avoid accidentally comparing stale results. The cache is copied into the direct run so no EPA API call is needed for already-cached DTXSIDs. The Tox21 inference service **is** called again and must be available. A nonzero exit status means a mismatch (1) or verification error (2).

This tests deterministic stage outputs, not full model training or OAF restart/recovery. If the Tox21 service is nondeterministic, its scoring stage may legitimately differ; inspect the JSON report before drawing conclusions. `NOT_COMPARABLE` in the separate legacy regression audit is not itself an OAF/direct mismatch.
