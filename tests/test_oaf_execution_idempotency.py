import pytest
from biotoxbridge.oaf.agent import execute_stage
from biotoxbridge.oaf.artifacts import publish_json, read_json


def test_replay_skips_scientific_function(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    calls = []
    def stage(data):
        calls.append(data)
        return {"x": data["x"] * 2}
    kwargs = dict(root=tmp_path, run_id="r1", stage="double", input_ref=source,
                  stages={"double": stage}, execution_id="evt1")
    first = execute_stage(**kwargs)
    assert execute_stage(**kwargs) == first
    assert len(calls) == 1
    assert read_json(first, tmp_path) == {"x": 4}


def test_replay_validates_input_before_reuse(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    kwargs = dict(root=tmp_path, run_id="r1", stage="double", input_ref=source,
                  stages={"double": lambda d: d}, execution_id="evt1")
    execute_stage(**kwargs)
    from pathlib import Path
    Path(source["uri"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        execute_stage(**kwargs)


def test_invalid_json_output_is_not_reused(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    kwargs = dict(root=tmp_path, run_id="r1", stage="double", input_ref=source,
                  stages={"double": lambda d: d}, execution_id="evt1")
    ref = execute_stage(**kwargs)
    from pathlib import Path
    Path(ref["uri"]).write_text("not-json")
    with pytest.raises(ValueError):
        execute_stage(**kwargs)


def test_missing_output_executes(tmp_path):
    source = publish_json(tmp_path, "r1", "input", {"x": 2})
    calls = []
    def stage(d):
        calls.append(1)
        return d
    execute_stage(root=tmp_path, run_id="r1", stage="double", input_ref=source,
                  stages={"double": stage}, execution_id="evt1")
    assert calls == [1]


def test_changed_input_same_event_is_rejected(tmp_path):
    a = publish_json(tmp_path, "r1", "a", {"x": 1})
    b = publish_json(tmp_path, "r1", "b", {"x": 2})
    kw = dict(root=tmp_path, run_id="r1", stage="double", stages={"double": lambda d: d}, execution_id="evt1")
    execute_stage(input_ref=a, **kw)
    with pytest.raises(ValueError, match="Artifact conflict"):
        execute_stage(input_ref=b, **kw)

def test_binding_conflict_prevents_scientific_execution(tmp_path):
    a = publish_json(tmp_path, "r1", "a", {"x": 1})
    b = publish_json(tmp_path, "r1", "b", {"x": 2})
    calls = []
    kw = dict(root=tmp_path, run_id="r1", stage="double", stages={"double": lambda d: calls.append(d) or d}, execution_id="evt1")
    execute_stage(input_ref=a, **kw)
    with pytest.raises(ValueError, match="Artifact conflict"):
        execute_stage(input_ref=b, **kw)
    assert len(calls) == 1
