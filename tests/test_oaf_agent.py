import asyncio
import sys
import types
from uuid import uuid4

import pytest

from biotoxbridge.oaf.agent import (EVENT_COMPLETED, EVENT_REQUESTED,
                                    create_agent_class, execute_stage)
from biotoxbridge.oaf.artifacts import publish_json, read_json


def test_stage_execution(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    out = execute_stage(root=tmp_path, run_id="r1", stage="double",
                        input_ref=source, stages={"double": lambda x: {"x": x["x"] * 2}},
                        execution_id="abc")
    assert read_json(out, tmp_path) == {"x": 4}


def test_stage_replay(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    kw = dict(root=tmp_path, run_id="r1", stage="double", input_ref=source,
              stages={"double": lambda x: {"x": 4}}, execution_id="abc")
    assert execute_stage(**kw) == execute_stage(**kw)


def test_stage_conflicting_replay(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    kw = dict(root=tmp_path, run_id="r1", stage="double",
              input_ref=source, execution_id="abc")

    first = execute_stage(
        **kw, stages={"double": lambda x: {"x": 4}}
    )
    replay = execute_stage(
        **kw, stages={"double": lambda x: {"x": 5}}
    )

    assert replay == first
    assert read_json(replay, tmp_path) == {"x": 4}


def test_unknown_stage(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {})
    with pytest.raises(ValueError, match="Unknown"):
        execute_stage(root=tmp_path, run_id="r1", stage="bad", input_ref=source,
                      stages={}, execution_id="abc")


def test_corrupt_input(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {})
    from pathlib import Path
    Path(source["uri"]).write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        execute_stage(root=tmp_path, run_id="r1", stage="ok", input_ref=source,
                      stages={"ok": lambda x: x}, execution_id="abc")


def test_oaf_envelope_contract_without_redis(tmp_path, monkeypatch):
    """Exercise actual adapter logic with lightweight OAF import-boundary stubs."""
    pkg = types.ModuleType("shared")
    pkg.__path__ = []
    agent_pkg = types.ModuleType("shared.agent")
    agent_pkg.__path__ = []
    agent_mod = types.ModuleType("shared.agent.base")
    class FakeBaseAgent:
        def __init__(self, **kw):
            self.name = kw["name"]
            self.subscribed_event_types = kw["subscribed_event_types"]
    agent_mod.BaseAgent = FakeBaseAgent
    event_pkg = types.ModuleType("shared.eventbus")
    event_pkg.__path__ = []
    event_mod = types.ModuleType("shared.eventbus.events")
    class FakeEnvelope:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    event_mod.EventEnvelope = FakeEnvelope
    for name, module in [("shared", pkg), ("shared.agent", agent_pkg),
                         ("shared.agent.base", agent_mod),
                         ("shared.eventbus", event_pkg),
                         ("shared.eventbus.events", event_mod)]:
        monkeypatch.setitem(sys.modules, name, module)
    Agent = create_agent_class()
    agent = Agent(event_bus=None, correlation_registry=None,
                  artifact_root=tmp_path, stages={"double": lambda x: x * 2})
    source = publish_json(tmp_path, "r1", "input", 3)
    event = FakeEnvelope(event_id=uuid4(), correlation_id=uuid4(),
                         event_type=EVENT_REQUESTED,
                         payload={"run_id": "r1", "stage": "double", "input_artifact": source})
    downstream = asyncio.run(agent.handle_event(event))
    assert len(downstream) == 1
    assert downstream[0].event_type == EVENT_COMPLETED
    assert downstream[0].correlation_id == event.correlation_id
    assert downstream[0].causation_id == event.event_id
    assert downstream[0].source_agent == "biotoxbridge"
    assert read_json(downstream[0].payload["output_artifact"], tmp_path) == 6
