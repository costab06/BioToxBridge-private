"""OAF BaseAgent bridge. No changes to OAF or the standalone pipeline.

The caller supplies a mapping of stage names to pure, synchronous functions.
Each function accepts one JSON-compatible value and returns one.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Mapping

from biotoxbridge.oaf.artifacts import publish_json, read_json

Stage = Callable[[Any], Any]
EVENT_REQUESTED = "biotoxbridge.stage.requested"
EVENT_COMPLETED = "biotoxbridge.stage.completed"


def execute_stage(*, root: str | Path, run_id: str, stage: str,
                  input_ref: dict, stages: Mapping[str, Stage],
                  execution_id: str) -> dict:
    """Verify input, execute an allowlisted stage and persist immutable output.

    A stable execution_id ensures redelivery resolves to the same artifact.
    """
    if stage not in stages:
        raise ValueError(f"Unknown BioToxBridge stage: {stage}")
    data = read_json(input_ref, root)
    result = stages[stage](data)
    return publish_json(root, run_id, f"{stage}-{execution_id}", result)


def create_agent_class():
    """Load OAF only when an OAF worker is actually constructed.

    Add OpenAgentFabric/framework to PYTHONPATH in the worker environment.
    """
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
            # Run potentially blocking scientific work outside the event loop.
            # OAF manages retries, leases and event publication after return.
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
