from pathlib import Path
from unittest.mock import Mock

import pytest
from biotoxbridge.oaf.scientific import scientific_stages
from biotoxbridge import stages


def test_all_scientific_stages_preserve_data_and_paths(tmp_path, monkeypatch):
    calls = {}
    def target(cfg, audit):
        calls['target'] = (cfg, audit)
        return [{'dtxsid': 'A'}]
    def resolve(rows, interim, audit):
        calls['resolve'] = (rows, interim, audit)
        return ([{'dtxsid': 'A'}], [{'dtxsid': 'B'}])
    def group(rows, cfg):
        return ([{'dtxsid': 'A', 'structure_group': 'G'}], [{'dtxsid': 'C'}])
    def score(rows, cfg, audit):
        return [{'dtxsid': 'A', 'tox21_x': -0.2}]
    def master(target, scored, processed, audit):
        calls['master'] = (target, scored, processed, audit)
        return ([{'dtxsid': 'A'}], [{'dtxsid': 'B'}])
    def split(rows, cfg, splits):
        calls['split'] = splits
        return ([rows[0]], [])
    for name, fn in [('build_toxicity_target', target), ('resolve_chemistry', resolve),
                     ('group_structures', group), ('score_tox21', score),
                     ('assemble_master', master), ('freeze_split', split)]:
        monkeypatch.setattr(stages, name, fn)
    cfg = {'experiment_id': 'test'}
    functions = scientific_stages(cfg, tmp_path)
    data = {}
    for name in ('toxicity_target', 'chemistry_resolution', 'structure_grouping',
                 'tox21_scoring', 'master_assembly', 'dataset_split'):
        data = functions[name](data)
    assert data['train'] == [{'dtxsid': 'A'}]
    assert data['excluded'] == [{'dtxsid': 'B'}]
    assert calls['resolve'][1] == tmp_path / 'interim'
    assert calls['master'][2] == tmp_path / 'processed'
    assert calls['split'] == tmp_path / 'splits'
    assert all((tmp_path / p).is_dir() for p in ('audit','interim','processed','splits'))


def test_missing_input_rejected(tmp_path):
    functions = scientific_stages({}, tmp_path)
    with pytest.raises(ValueError, match='Missing scientific input: target'):
        functions['chemistry_resolution']({})


def test_nonmapping_rejected(tmp_path):
    functions = scientific_stages({}, tmp_path)
    with pytest.raises(ValueError, match='JSON object'):
        functions['toxicity_target']([])


def test_empty_master_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, 'assemble_master', lambda *args: ([], []))
    functions = scientific_stages({}, tmp_path)
    with pytest.raises(ValueError, match='Empty master'):
        functions['master_assembly']({'target': [], 'scored': []})


def test_stage_registry_is_allowlisted(tmp_path):
    functions = scientific_stages({}, tmp_path)

    assert set(functions) == {
        "toxicity_target",
        "chemistry_resolution",
        "structure_grouping",
        "tox21_scoring",
        "master_assembly",
        "dataset_split",
        "model_training",
        "model_evaluation",
        "experiment_reporting",
    }
