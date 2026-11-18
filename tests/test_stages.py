"""Contracts for independently callable preparation stages."""
from biotoxbridge import stages


def test_target_stage_passes_config(monkeypatch, tmp_path):
    import biotoxbridge.targets.toxref_target as module
    calls = {}
    def fake_build_target(**kwargs):
        calls.update(kwargs)
        return [{"dtxsid": "DTXSID1"}]
    monkeypatch.setattr(module, "build_target", fake_build_target)
    cfg = {"target": {
        "raw_file": "raw.csv", "effect_substring": "liver",
        "species": "rat", "route": "oral",
        "study_types": ["CHR"], "duration_preference": ["CHR"],
        "pod_type": "LOAEL",
    }}
    assert stages.build_toxicity_target(cfg, tmp_path) == [{"dtxsid": "DTXSID1"}]
    assert calls["audit_csv"] == tmp_path / "toxref_target.csv"
    assert calls["study_types"] == ("CHR",)


def test_split_stage_delegates_to_existing_validation(monkeypatch, tmp_path):
    import biotoxbridge.splitting.splitter as module
    calls = {}
    def fake_make(master, **kwargs):
        calls["make"] = kwargs
    def fake_load(master, path):
        calls["load"] = path
        return master[:1], master[1:]
    monkeypatch.setattr(module, "make_split", fake_make)
    monkeypatch.setattr(module, "load_split", fake_load)
    cfg = {"split": {"test_fraction": 0.2, "seed": 42, "group_by": "structure_group"}}
    train, test = stages.freeze_split([{"dtxsid": "1"}, {"dtxsid": "2"}], cfg, tmp_path)
    assert len(train) == len(test) == 1
    assert calls["load"] == tmp_path / "split_manifest.json"
    assert calls["make"]["seed"] == 42
