from biotoxbridge.oaf.esol_lifecycle_orchestration import next_request


def test_transitions():
    cfg = {'seed': 42, 'policy': {'max_mae': .1, 'min_r2': .99, 'max_mae_regression': 0}}
    assert next_request('esol_acquire', {'sha256': 'abc'}, cfg) == ('esol_train', {'dataset_sha256': 'abc', 'seed': 42})
    assert next_request('esol_train', {'candidate_version': 'v1'}, cfg) == ('esol_promote', {'candidate_version': 'v1', **cfg['policy']})
    assert next_request('esol_promote', {'accepted': False}, cfg) == (None, None)
