import pytest
from biotoxbridge.oaf.modeling import finite_json, model_ids, report_models


def test_model_mapping():
    assert model_ids({'models': ['structure_only', 'predicted_bioactivity', 'structure_plus_predicted']}) == ['A', 'B', 'D']
    with pytest.raises(ValueError):
        model_ids({'models': ['A', 'A']})
    with pytest.raises(ValueError):
        model_ids({'models': ['unknown']})


def test_nonfinite_statistics_are_json_safe():
    import json
    assert json.dumps(finite_json({'rho': float('nan'), 'r2': float('-inf')}), allow_nan=False) == '{"rho": null, "r2": null}'


def test_report_files(tmp_path):
    value = {'train_count': 2, 'test_count': 1, 'excluded_count': 0,
             'models': {'A': {'predictions': [{'dtxsid': 'X', 'y_true': 1.0, 'y_pred': 0.5, 'residual': 0.5}]}},
             'metrics': {'A': {'mae': 0.5}}, 'comparison': {'mae': {'A': 0.5}}}
    result = report_models(value, tmp_path)
    assert result['test_count'] == 1
    assert (tmp_path / 'metrics.json').exists()
    assert 'X,A,1.0,0.5,0.5' in (tmp_path / 'predictions.csv').read_text()
