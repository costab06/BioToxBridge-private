# OAF recovery contract — diagnostic tests v1

Unzip from the BioToxBridge project root and run:

```bash
python -m pytest -v tests/test_oaf_recovery_contract.py
```

The test suite checks checksum corruption detection, immutable artifact conflict behavior, and duplicate delivery. One test may show **XFAIL**: it deliberately exposes the existing limitation that a repeated event can recompute a scientific stage even though the immutable artifact filename is stable. **Do not treat XFAIL as a passing recovery guarantee.**

These tests do not simulate Redis consumer recovery, SIGKILL, lease expiry, or DLQ redrive. Those require a separate controlled end-to-end test with Redis 7 and a worker process. Do not interrupt a real chemistry/Tox21 run until a crash-safe execution plan is in place: `asyncio.to_thread` work may continue after cancellation and some stages write output files before their immutable handoff artifact is committed.

No OAF source code is modified.
