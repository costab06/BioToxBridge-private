import importlib.util
import sys
import types

for name in ("biotoxbridge", "biotoxbridge.oaf", "biotoxbridge.oaf.artifacts", "biotoxbridge.oaf.scientific"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["biotoxbridge.oaf.artifacts"].read_json = lambda *a: None
sys.modules["biotoxbridge.oaf.scientific"].scientific_stages = lambda *a, **k: {}
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'scripts/verify_oaf_equivalence.py'
spec = importlib.util.spec_from_file_location('equivalence', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_optional_metadata_only_on_rejected_records():
    direct = {'missing': [{'dtxsid': 'A', 'failure_reason': 'qsarReadySmiles_missing', 'inchiString': '', 'inchikey': '', 'msReadySmiles': ''}]}
    oaf = {'missing': [{'dtxsid': 'A', 'failure_reason': 'qsarReadySmiles_missing', 'inchiString': None, 'inchikey': None, 'msReadySmiles': None}]}
    differences, notes = module.compare_stage(direct, oaf)
    assert not differences
    assert notes[0]['direct_empty_fields'] == 3


def test_identifiers_remain_strict():
    d = {'missing': [{'dtxsid': '', 'failure_reason': 'qsarReadySmiles_missing'}]}
    o = {'missing': [{'dtxsid': None, 'failure_reason': 'qsarReadySmiles_missing'}]}
    assert module.compare_stage(d, o)[0]


def test_non_missing_records_remain_strict():
    d = {'missing': [{'dtxsid': 'A', 'inchiString': ''}]}
    o = {'missing': [{'dtxsid': 'A', 'inchiString': None}]}
    assert module.compare_stage(d, o)[0]


def test_scores_remain_strict():
    assert module.compare_stage({'scored': [{'score': 0.2}]}, {'scored': [{'score': 0.20000001}]})[0]


def test_missing_dtxcid_and_smiles_are_normalized_only_when_rejected():
    d = {'missing': [{'dtxsid': 'DTXSID0048056', 'failure_reason': 'qsarReadySmiles_missing', 'dtxcid': '', 'smiles': ''}]}
    o = {'missing': [{'dtxsid': 'DTXSID0048056', 'failure_reason': 'qsarReadySmiles_missing', 'dtxcid': None, 'smiles': None}]}
    differences, notes = module.compare_stage(d, o)
    assert not differences
    assert notes[0]['direct_empty_fields'] == 2


def test_accepted_smiles_and_dtxcid_remain_strict():
    d = {'resolved': [{'dtxsid': 'A', 'dtxcid': '', 'smiles': ''}]}
    o = {'resolved': [{'dtxsid': 'A', 'dtxcid': None, 'smiles': None}]}
    assert module.compare_stage(d, o)[0]
