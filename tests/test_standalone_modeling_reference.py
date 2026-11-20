import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_standalone_modeling_reference.py'
spec = importlib.util.spec_from_file_location('standalone_modeling_reference', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_find_split_excludes_binding(tmp_path):
    run = tmp_path / 'example'
    run.mkdir()
    (run / 'dataset_split-abc-input-binding.json').write_text('{}')
    output = run / 'dataset_split-abc.json'
    output.write_text('{}')
    assert module.find_split(tmp_path, 'example') == output


def test_find_split_requires_unique_output(tmp_path):
    run = tmp_path / 'example'
    run.mkdir()
    with pytest.raises(ValueError, match='found 0'):
        module.find_split(tmp_path, 'example')
    (run / 'dataset_split-a.json').write_text('{}')
    (run / 'dataset_split-b.json').write_text('{}')
    with pytest.raises(ValueError, match='found 2'):
        module.find_split(tmp_path, 'example')
