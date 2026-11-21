import pytest
from biotoxbridge.oaf.artifacts import publish_json, read_json


def test_round_trip(tmp_path):
    ref = publish_json(tmp_path, "run-1", "target", [{"dtxsid": "DTXSID1"}])
    assert ref["artifact_type"] == "evaluation"
    assert read_json(ref, tmp_path) == [{"dtxsid": "DTXSID1"}]


def test_idempotent_replay(tmp_path):
    first = publish_json(tmp_path, "run-1", "target", {"n": 1})
    second = publish_json(tmp_path, "run-1", "target", {"n": 1})
    assert first == second


def test_conflicting_replay(tmp_path):
    publish_json(tmp_path, "run-1", "target", {"n": 1})
    with pytest.raises(ValueError, match="conflict"):
        publish_json(tmp_path, "run-1", "target", {"n": 2})


def test_corruption(tmp_path):
    ref = publish_json(tmp_path, "run-1", "target", {"n": 1})
    from pathlib import Path
    Path(ref["uri"]).write_text('{"n":2}')
    with pytest.raises(ValueError, match="checksum"):
        read_json(ref, tmp_path)


def test_reject_traversal(tmp_path):
    with pytest.raises(ValueError):
        publish_json(tmp_path, "../bad", "target", {})


def test_reject_reference_escape(tmp_path):
    ref = publish_json(tmp_path, "run-1", "target", {})
    ref["uri"] = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="path"):
        read_json(ref, tmp_path)


def test_reject_nan(tmp_path):
    with pytest.raises(ValueError):
        publish_json(tmp_path, "run-1", "target", {"x": float("nan")})
