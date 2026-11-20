# BioToxBridge OAF replay optimization v1

Changes only `src/biotoxbridge/oaf/agent.py`, adds `tests/test_oaf_execution_idempotency.py`.

Install from project root: `unzip -o ~/Downloads/biotoxbridge_oaf_execution_idempotency_v1.zip`.

Run `python -m pytest -q tests/test_oaf_execution_idempotency.py` and `python -m pytest -q`.

**Behavior change:** a second call with the same run/stage/execution ID returns the existing output without re-running the stage. Existing `test_stage_conflicting_replay` in `tests/test_oaf_agent.py` expects the old behavior and must be updated to assert reuse instead of conflict. The patch intentionally does not edit that test automatically.

**Limitations:** This does not establish exactly-once execution, bind an output to a cryptographically authenticated input or stage version, or detect tampering of a syntactically valid output whose checksum is reconstructed from the output itself. A persistent manifest with an independently stored checksum and input provenance is needed for stronger guarantees. Two concurrent workers can still execute the stage simultaneously. OAF is unchanged.
