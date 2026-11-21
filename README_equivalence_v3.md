# BioToxBridge OAF equivalence verifier v3

Extends the narrowly scoped empty-string/JSON-null normalization to `dtxcid` and `smiles` **only on records rejected with** `failure_reason=qsarReadySmiles_missing`. The existing three metadata fields remain normalized under the same condition. DTXSID, accepted structures, and all prediction values are compared strictly. No OAF artifacts are modified.

Install from BioToxBridge repository root:

```bash
unzip -o ~/Downloads/biotoxbridge_oaf_equivalence_v3.zip
python -m pytest -q tests/test_verify_oaf_equivalence.py
python scripts/verify_oaf_equivalence.py \
  --config config/experiments/liver_loael_v1.yaml \
  --run-id oaf-test-002 \
  --artifact-root runs/oaf-artifacts \
  --direct-work-dir runs/equivalence-direct-004 \
  --cache-csv runs/oaf-work/oaf-test-002/interim/dsstox_cache.csv \
  --report runs/equivalence-report-004.json
```

Requires the local Tox21 inference service.
