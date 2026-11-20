# ESOL Phase 3: OAF-compatible scientific stages and resumable inference

This patch is additive and **does not change OpenAgentFabric** or existing LOAEL stages.

## Tests

```bash
unzip -o ~/Downloads/biotoxbridge_esol_phase3_v1.zip
python -m pytest -q tests/test_esol_oaf_workflows.py
python -m pytest -q
```

## Local batch demonstration

Create `chemicals.json`:

```json
[{"id":"ethanol","smiles":"CCO"},{"id":"benzene","smiles":"c1ccccc1"}]
```

With the Tox21 HTTP service running:

```bash
python scripts/esol_oaf_batch.py --run-id esol-demo-001 --input chemicals.json
```

For solubility-only development:

```bash
python scripts/esol_oaf_batch.py --run-id esol-demo-local-001 --input chemicals.json --skip-tox21
```

Repeat the exact command to verify checkpoint reuse. The batch pins the accepted model
version in its manifest; resuming uses that pinned version even after promotion.
Individual successful predictions are immutable JSON artifacts.

## OAF integration boundary

`lifecycle_stages(store)` returns synchronous JSON-in/JSON-out stage functions
compatible with `biotoxbridge.oaf.agent.execute_stage` and its OAF BaseAgent wrapper.
A separate Redis event coordinator, scheduled acquisition/retraining, and automated
promotion policies are **not implemented in this increment**. This is a testable
bridge and local resumable inference demonstration, not yet a complete live OAF
multi-process lifecycle. Promotion remains an explicit action.

## Caveats

- The registry's promotion lock is still caller-managed; concurrent promotions
  require coordinator serialization.
- `joblib` loads trusted local artifacts only.
- The batch uses immutable JSON artifacts but is not an exactly-once external
  inference guarantee across crashes.
- Repeated batch runs must use the same input and options.
