"""Deterministic six-stage BioToxBridge workflow coordinator.

Pure transition logic. OAF worker calls transition() for completion events and
returns the resulting envelope(s); OAF owns publication, claims, and retries.
"""
from __future__ import annotations

from uuid import UUID, uuid5
from pathlib import Path

from biotoxbridge.oaf.agent import EVENT_COMPLETED, EVENT_REQUESTED
from biotoxbridge.oaf.artifacts import publish_json, read_json

STAGES = (
    "toxicity_target", "chemistry_resolution", "structure_grouping",
    "tox21_scoring", "master_assembly", "dataset_split",
    "model_training", "model_evaluation", "experiment_reporting",
)
EVENT_FINISHED = "biotoxbridge.workflow.finished"
EVENT_STARTED = "biotoxbridge.workflow.started"


def _request(event, stage: str, reference: dict, envelope_type):
    # Stable downstream event identity across retries of the same source.
    return envelope_type(
        event_id=uuid5(event.event_id, f"biotoxbridge:{stage}"),
        event_type=EVENT_REQUESTED,
        source_agent="biotoxbridge-coordinator",
        correlation_id=event.correlation_id,
        causation_id=event.event_id,
        payload={"run_id": event.payload["run_id"], "stage": stage,
                 "input_artifact": reference},
    )


def start(event, root: str | Path, envelope_type):
    """Convert a workflow.started event into the first stage request."""
    if event.event_type != EVENT_STARTED:
        raise ValueError("Expected workflow start")
    if not isinstance(event.payload, dict) or not isinstance(event.payload.get("run_id"), str):
        raise ValueError("Missing run_id")
    run_id = event.payload["run_id"]
    # The initial input is empty; config belongs to the server-side stage registry.
    ref = publish_json(root, run_id, "initial-input", {})
    return [_request(event, STAGES[0], ref, envelope_type)]


def transition(event, root: str | Path, envelope_type):
    """Validate a stage completion and return the next stage request or finish."""
    if event.event_type != EVENT_COMPLETED:
        raise ValueError("Expected stage completion")
    payload = event.payload
    if not isinstance(payload, dict):
        raise ValueError("Completion payload must be a mapping")
    run_id, stage, ref = (payload.get(k) for k in ("run_id", "stage", "output_artifact"))
    if not isinstance(run_id, str) or stage not in STAGES or not isinstance(ref, dict):
        raise ValueError("Invalid stage completion")
    artifact_id = ref.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.startswith(run_id + "/" + stage + "-"):
        raise ValueError("Stage output artifact does not match run and stage")
    # Verify integrity BEFORE emitting any downstream event.
    result = read_json(ref, root)
    if not isinstance(result, dict):
        raise ValueError("Scientific stage output must be a mapping")
    index = STAGES.index(stage)
    if index + 1 < len(STAGES):
        return [_request(event, STAGES[index + 1], ref, envelope_type)]
    if not isinstance(result.get("metrics"), dict) or not isinstance(result.get("comparison"), dict):
        raise ValueError("Final report must contain metrics and comparison")
    if not isinstance(result.get("train_count"), int) or not isinstance(result.get("test_count"), int):
        raise ValueError("Final report must contain train/test counts")
    return [envelope_type(
        event_id=uuid5(event.event_id, "biotoxbridge:finished"),
        event_type=EVENT_FINISHED,
        source_agent="biotoxbridge-coordinator",
        correlation_id=event.correlation_id,
        causation_id=event.event_id,
        payload={"run_id": run_id, "output_artifact": ref,
                 "train_count": result["train_count"], "test_count": result["test_count"]},
    )]


def create_coordinator_class():
    """Bind pure transitions to the installed OAF BaseAgent."""
    from shared.agent.base import BaseAgent
    from shared.eventbus.events import EventEnvelope

    class BioToxBridgeCoordinator(BaseAgent):
        def __init__(self, *, event_bus, correlation_registry, artifact_root, **kwargs):
            super().__init__(
                name="biotoxbridge-coordinator",
                event_bus=event_bus,
                correlation_registry=correlation_registry,
                subscribed_event_types={EVENT_STARTED, EVENT_COMPLETED},
                completion_event_type=EVENT_FINISHED,
                **kwargs,
            )
            self.artifact_root = Path(artifact_root)

        async def handle_event(self, event):
            if event.event_type == EVENT_STARTED:
                return start(event, self.artifact_root, EventEnvelope)
            return transition(event, self.artifact_root, EventEnvelope)

    return BioToxBridgeCoordinator
