"""Recovery contract probes for BioToxBridge's OAF adapter.

Run: python -m pytest -v tests/test_oaf_recovery_contract.py
These are local adapter tests, not Redis/worker-crash integration tests.
"""
import json
from pathlib import Path

import pytest

from biotoxbridge.oaf.agent import execute_stage
from biotoxbridge.oaf.artifacts import publish_json, read_json


def test_duplicate_delivery_reuses_identical_artifact(tmp_path):
    """A replay must not change an existing artifact's bytes or identity."""
    initial = publish_json(tmp_path, "run-1", "input", {"value": 3})
    calls = []

    def stage(value):
        calls.append(1)
        return {"value": value["value"] * 2}

    kwargs = dict(root=tmp_path, run_id="run-1", stage="sample",
                  input_ref=initial, stages={"sample": stage}, execution_id="event-1")
    first = execute_stage(**kwargs)
    second = execute_stage(**kwargs)
    assert first == second
    assert read_json(first, tmp_path) == {"value": 6}
    # Diagnostic: the current adapter may recompute on replay. This is a
    # documented limitation, NOT proof of exactly-once stage execution.
    if len(calls) != 1:
        pytest.xfail("Stage was executed more than once on duplicate delivery; "
                     "artifact idempotency does not imply execution idempotency")


def test_duplicate_delivery_reuses_existing_result(tmp_path):
    """Duplicate delivery must reuse the output without rerunning the stage."""
    initial = publish_json(tmp_path, "run-1", "input", {})
    count = 0

    def stage(_):
        nonlocal count
        count += 1
        return {"attempt": count}

    kwargs = dict(
        root=tmp_path,
        run_id="run-1",
        stage="sample",
        input_ref=initial,
        stages={"sample": stage},
        execution_id="event-1",
    )

    first = execute_stage(**kwargs)
    second = execute_stage(**kwargs)

    assert second == first
    assert count == 1
    assert read_json(second, tmp_path) == {"attempt": 1}

def test_corrupt_artifact_is_detected(tmp_path):
    ref = publish_json(tmp_path, "run-1", "sample", {"value": 3})
    Path(ref["uri"]).write_text('{"value":4}\n')
    with pytest.raises(ValueError, match="checksum"):
        read_json(ref, tmp_path)


def test_changed_input_with_same_event_identity_conflicts(tmp_path):
    """A replay with the same event identity cannot silently replace output."""
    a = publish_json(tmp_path, "run-1", "a", {"value": 1})
    b = publish_json(tmp_path, "run-1", "b", {"value": 2})
    stages = {"sample": lambda d: d}
    args = dict(root=tmp_path, run_id="run-1", stage="sample",
                stages=stages, execution_id="event-1")
    execute_stage(input_ref=a, **args)
    with pytest.raises(ValueError, match="Artifact conflict"):
        execute_stage(input_ref=b, **args)
