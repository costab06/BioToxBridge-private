from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from biotoxbridge.oaf.coordinator import (
    STAGES, EVENT_STARTED, EVENT_FINISHED, start, transition,
)
from biotoxbridge.oaf.agent import EVENT_COMPLETED, EVENT_REQUESTED
from biotoxbridge.oaf.artifacts import publish_json


def envelope(**kwargs):
    return SimpleNamespace(**kwargs)


def event(event_type, payload):
    return envelope(event_id=uuid4(), correlation_id=uuid4(),
                    event_type=event_type, payload=payload)


def test_full_sequence(tmp_path):
    original = event(EVENT_STARTED, {"run_id": "run1"})
    request = start(original, tmp_path, envelope)[0]

    assert request.event_type == EVENT_REQUESTED

    for stage in STAGES:
        assert request.payload["stage"] == stage

        if stage == "dataset_split":
            data = {"train": [1], "test": [2]}
        elif stage == "experiment_reporting":
            data = {
                "metrics": {"A": {"mae": 0.5}},
                "comparison": {"mae": {"A": 0.5}},
                "train_count": 1,
                "test_count": 1,
            }
        else:
            data = {"stage": stage}

        output = publish_json(
            tmp_path, "run1", stage + "-123", data
        )
        completed = event(
            EVENT_COMPLETED,
            {
                "run_id": "run1",
                "stage": stage,
                "output_artifact": output,
            },
        )
        completed.correlation_id = request.correlation_id

        request = transition(completed, tmp_path, envelope)[0]

        assert request.causation_id == completed.event_id
        assert request.correlation_id == completed.correlation_id

    assert request.event_type == EVENT_FINISHED
    assert request.payload["train_count"] == 1

def test_replay_deterministic(tmp_path):
    e = event(EVENT_STARTED, {"run_id": "r"})
    assert start(e, tmp_path, envelope)[0].event_id == start(e, tmp_path, envelope)[0].event_id


def test_corrupt_artifact_rejected(tmp_path):
    ref = publish_json(tmp_path, "r", "toxicity_target-1", {})
    Path(ref["uri"]).write_text("bad")
    e = event(EVENT_COMPLETED, {"run_id": "r", "stage": STAGES[0], "output_artifact": ref})
    with pytest.raises(ValueError, match="checksum"):
        transition(e, tmp_path, envelope)


def test_cross_run_artifact_rejected(tmp_path):
    ref = publish_json(tmp_path, "other", "toxicity_target-1", {})
    e = event(EVENT_COMPLETED, {"run_id": "r", "stage": STAGES[0], "output_artifact": ref})
    with pytest.raises(ValueError, match="does not match"):
        transition(e, tmp_path, envelope)


def test_unknown_stage_rejected(tmp_path):
    ref = publish_json(tmp_path, "r", "unexpected-1", {})
    e = event(EVENT_COMPLETED, {"run_id": "r", "stage": "unexpected", "output_artifact": ref})
    with pytest.raises(ValueError, match="Invalid stage"):
        transition(e, tmp_path, envelope)


def test_final_report_validation(tmp_path):
    ref = publish_json(
        tmp_path,
        "r",
        "experiment_reporting-1",
        {"metrics": {}},
    )

    e = event(
        EVENT_COMPLETED,
        {
            "run_id": "r",
            "stage": "experiment_reporting",
            "output_artifact": ref,
        },
    )

    with pytest.raises(
        ValueError,
        match="metrics and comparison",):
        transition(e, tmp_path, envelope)
