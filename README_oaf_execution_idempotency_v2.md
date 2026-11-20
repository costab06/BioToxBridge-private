# Execution idempotency v2

Changes `src/biotoxbridge/oaf/agent.py` to persist an immutable input binding before execution. Different input references with the same run/stage/event ID raise `Artifact conflict` before scientific work. Existing output is reused for matching input.

**Install:** `unzip -o ~/Downloads/biotoxbridge_oaf_execution_idempotency_v2.zip`

**Tests:** `python -m pytest -q tests/test_oaf_execution_idempotency.py`

**Update three legacy tests:**
- `tests/test_oaf_agent.py::test_stage_conflicting_replay`: replace the `pytest.raises` expectation with `assert execute_stage(**kw, stages={"double": lambda x: {"x": 5}}) == first_ref` where `first_ref` is the result of the initial call. A replay with the same input skips the stage.
- `tests/test_oaf_recovery_contract.py::test_duplicate_delivery_changed_result_is_rejected`: replace the raises block with `assert execute_stage(**kwargs) == ref; assert count == 1`.
- `tests/test_oaf_recovery_contract.py::test_changed_input_with_same_event_identity_conflicts`: retain the existing `pytest.raises(ValueError, match="Artifact conflict")` block; v2 makes it pass.

**Compatibility:** v1 outputs lack input-binding artifacts. For runs created under v1, a replay will create a binding for the supplied input but cannot independently prove it matches the original. Use a fresh run ID for recovery tests.

**Limitations:** This is not exactly-once execution. Two workers can race before output publication; crash before publication can repeat side effects. A separately trusted output digest and worker interruption tests remain pending.
