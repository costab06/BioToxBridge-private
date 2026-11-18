# BioToxBridge / OpenAgentFabric integration — Phase 1

OpenAgentFabric is **not modified** in this deliverable.

## Implemented
- Every pipeline execution gets a unique run directory for processed datasets, split manifests, metrics, predictions and audits.
- Frozen split loading rejects changed chemical populations, duplicates, group leakage and inconsistent counts.
- Master assembly rejects duplicate scored/target DTXSIDs rather than silently overwriting.
- Added regression tests for these failure modes.

## Important limitations (not yet implemented)
- `--reuse-split` now fails explicitly instead of silently reading an ambiguous shared split. A future CLI option should accept an explicit prior manifest and validate it against the current dataset.
- Source cache in `data/interim` is still shared; the cache must be versioned or snapshotted before restart-safe OAF execution.
- Model and scaler serialization is not yet implemented.
- OAF event handlers, stage contracts, fenced artifact publication, and crash recovery tests are not yet implemented.
- No end-to-end scientific experiment was run; external EPA inputs and NN inference service are required.

## Next
Refactor `pipeline.py` into pure stage functions with explicit artifact manifests, preserving scientific behavior; then build OAF adapters solely inside BioToxBridge.

## Phase 2a — scientific preparation stage extraction
- Added `src/biotoxbridge/stages.py` with six independently callable preparation stages.
- `pipeline.py` invokes these functions, retaining original stage order and scientific libraries.
- Added delegation contract tests.
- This is a preparation refactor, not yet durable execution: no checkpoint reuse, atomic artifact publication, OAF event handlers, or model persistence.
- End-to-end scientific validation still requires EPA inputs and the inference service.
- OpenAgentFabric is unchanged.
