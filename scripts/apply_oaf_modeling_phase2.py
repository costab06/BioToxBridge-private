"""Patch existing OAF stage registry/coordinator without replacing local v2 changes."""
from pathlib import Path

scientific = Path('src/biotoxbridge/oaf/scientific.py')
coordinator = Path('src/biotoxbridge/oaf/coordinator.py')
for p in (scientific, coordinator):
    if not p.is_file():
        raise SystemExit(f'Missing {p}; run from BioToxBridge repository root')

s = scientific.read_text()
if "'model_training': model_training" not in s:
    needle = "    return {\n        'toxicity_target': target,"
    assert s.count(needle) == 1, 'Unexpected scientific.py layout; no changes made'
    block = '''    from biotoxbridge.oaf.modeling import train_models, evaluate_models, report_models

    def model_training(value: Any) -> dict:
        value = mapping(value, 'train', 'test')
        return train_models(value, config)

    def model_evaluation(value: Any) -> dict:
        value = mapping(value, 'models', 'train_count', 'test_count')
        return evaluate_models(value, config)

    def experiment_reporting(value: Any) -> dict:
        value = mapping(value, 'models', 'metrics', 'comparison')
        return report_models(value, run_dir)

'''
    s = s.replace(needle, block + needle)
    needle2 = "        'dataset_split': split,"
    assert s.count(needle2) == 1
    s = s.replace(needle2, needle2 + "\n        'model_training': model_training,\n        'model_evaluation': model_evaluation,\n        'experiment_reporting': experiment_reporting,")

c = coordinator.read_text()
if '"model_training"' not in c:
    needle = '"master_assembly", "dataset_split",'
    assert c.count(needle) == 1, 'Unexpected coordinator.py STAGES; no changes made'
    c = c.replace(needle, needle + '\n    "model_training", "model_evaluation", "experiment_reporting",')
    # Final reporting result has counts and metrics, not train/test record arrays.
    needle2 = '''    if not isinstance(result.get("train"), list) or not isinstance(result.get("test"), list):
        raise ValueError("Final split must contain train and test lists")'''
    assert c.count(needle2) == 1, 'Unexpected coordinator final validation; no changes made'
    c = c.replace(needle2, '''    if not isinstance(result.get("metrics"), dict) or not isinstance(result.get("comparison"), dict):
        raise ValueError("Final report must contain metrics and comparison")
    if not isinstance(result.get("train_count"), int) or not isinstance(result.get("test_count"), int):
        raise ValueError("Final report must contain train/test counts")''')
    c = c.replace('len(result["train"]), "test_count": len(result["test"])',
                  'result["train_count"], "test_count": result["test_count"]')
scientific.write_text(s)
coordinator.write_text(c)
print('Phase 2 OAF stage registry and coordinator updated.')
