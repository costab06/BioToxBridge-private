import pytest
from biotoxbridge.oaf.runtime import config_fingerprint, load_configuration, run_paths


def test_fingerprint_order_independent():
    assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})


def test_fingerprint_changes_with_config():
    assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 2})


def test_bad_run_id(tmp_path):
    with pytest.raises(ValueError):
        run_paths("../escape", tmp_path, tmp_path)


def test_config_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("not-a-mapping")
    # YAML is deliberately a single string here; not a mapping.
    with pytest.raises(ValueError, match="mapping"):
        load_configuration(path)
