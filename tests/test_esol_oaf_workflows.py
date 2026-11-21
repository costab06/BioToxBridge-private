import pytest
from biotoxbridge.oaf.esol_workflows import lifecycle_stages, accepted_version, compact_status


def test_stage_names(tmp_path):
    stages = lifecycle_stages(tmp_path)
    assert set(stages) == {'esol_acquire', 'esol_train', 'esol_promote', 'esol_infer'}


def test_requires_accepted_model(tmp_path):
    with pytest.raises(RuntimeError, match='No accepted'):
        accepted_version(tmp_path)
    assert compact_status(tmp_path) == {'status': 'NO_ACCEPTED_MODEL'}


def test_inference_requires_pinned_version(tmp_path):
    with pytest.raises(KeyError):
        lifecycle_stages(tmp_path)['esol_infer']({'smiles': 'CCO', 'include_tox21': False})
