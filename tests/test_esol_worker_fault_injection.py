"""Unit-test one-shot marker without running a Redis worker."""
import importlib.util
from pathlib import Path


def test_claim_once(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'esol_oaf_worker.py'
    spec = importlib.util.spec_from_file_location('esol_oaf_worker_fault_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = tmp_path / 'nested' / 'once.marker'
    assert module.claim_once(marker)
    assert not module.claim_once(marker)
    assert marker.read_text() == 'injected\n'
