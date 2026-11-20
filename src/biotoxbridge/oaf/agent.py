"""OAF BaseAgent bridge. No changes to OAF or the standalone pipeline.

The caller supplies a mapping of stage names to synchronous functions.
Each function accepts one JSON-compatible value and returns one.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Mapping

from biotoxbridge.oaf.artifacts import publish_json, read_json, _safe_path

Stage = Callable[[Any], Any]
EVENT_REQUESTED = "biotoxbridge.stage.requested"
EVENT_COMPLETED = "biotoxbridge.stage.completed"


def _existing_output(root: str | Path, run_id: str, stage: str, execution_id: str) -> dict | None:
    """Return a checksum-validated reference for an already published output.

    A corrupt existing artifact is an error, never permission to overwrite it.
    """
    root = Path(root).resolve()
    path = _safe_path(root, run_id, f"{stage}-{execution_id}")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Existing output is not a regular file: {path}")
    import hashlib
    payload = path.read_bytes()
    reference = {
        "artifact_id": f"{run_id}/{stage}-{execution_id}.json",
        "artifact_type": "evaluation",
        "uri": str(path),
        "version": 1,
        "checksum": hashlib.sha256(payload).hexdigest(),
    }
    # The checksum above establishes the reference, while read_json validates
    # identity/containment and JSON parsing. For end-to-end tamper detection,
    # persist an independent digest in a future manifest (see README).
    read_json(reference, root)
    return reference


def execute_stage(*, root: str | Path, run_id: str, stage: str,
                  input_ref: dict, stages: Mapping[str, Stage],
                  execution_id: str) -> dict:
    """Validate input and reuse an existing output for the same event identity.

    This is a replay optimization, not a distributed exactly-once guarantee.
    """
    if stage not in stages:
        raise ValueError(f"Unknown BioToxBridge stage: {stage}")
    data = read_json(input_ref, root)
    # Bind the event identity to its original input before doing any work.
    # Immutable publication rejects an event ID reused with a different input.
    # This metadata is intentionally separate from the scientific output.
    binding_stage = f"{stage}-{execution_id}-input-binding"
    binding = {
        "input_artifact_id": input_ref["artifact_id"],
        "input_checksum": input_ref["checksum"],
    }
    publish_json(root, run_id, binding_stage, binding)
    existing = _existing_output(root, run_id, stage, execution_id)
    if existing is not None:
        return existing
    result = stages[stage](data)
    return publish_json(root, run_id, f"{stage}-{execution_id}", result)


def create_agent_class():
    """Load OAF only when an OAF worker is constructed."""
    from shared.agent.base import BaseAgent
    from shared.eventbus.events import EventEnvelope

    class BioToxBridgeAgent(BaseAgent):
        def __init__(self, *, event_bus, correlation_registry, artifact_root,
                     stages: Mapping[str, Stage], name="biotoxbridge", **kwargs):
            super().__init__(
                name=name, event_bus=event_bus,
                correlation_registry=correlation_registry,
                subscribed_event_types={EVENT_REQUESTED}, **kwargs,
            )
            self.artifact_root = Path(artifact_root)
            self.stages = dict(stages)

        async def handle_event(self, event):
            if event.event_type != EVENT_REQUESTED:
                raise ValueError(f"Unexpected event type: {event.event_type}")
            payload = event.payload
            if not isinstance(payload, dict):
                raise ValueError("Stage request payload must be a mapping")
            for key in ("run_id", "stage", "input_artifact"):
                if key not in payload:
                    raise ValueError(f"Missing stage request field: {key}")
            output = await asyncio.to_thread(
                execute_stage, root=self.artifact_root,
                run_id=payload["run_id"], stage=payload["stage"],
                input_ref=payload["input_artifact"], stages=self.stages,
                execution_id=event.event_id.hex,
            )
            return [EventEnvelope(
                event_type=EVENT_COMPLETED,
                source_agent=str(self.name),
                correlation_id=event.correlation_id,
                causation_id=event.event_id,
                payload={"run_id": payload["run_id"],
                         "stage": payload["stage"],
                         "output_artifact": output},
            )]

    return BioToxBridgeAgent
