import pytest
from biotoxbridge.solubility.lifecycle import consolidate_records


def test_canonical_duplicates_consolidated():
    rows = [{'smiles': 'CCO', 'measured_log_solubility': -1.0},
            {'smiles': 'OCC', 'measured_log_solubility': -1.2}]
    rows += [{'smiles': 'C' * n, 'measured_log_solubility': -float(n)}
             for n in range(1, 22)]
    records, stats = consolidate_records(rows)
    assert len(records) == 22
    assert stats['duplicate_rows'] == 1
    assert stats['conflicting_label_structures'] == 1
    ethanol = next(r for r in records if r['smiles'] == 'CCO')
    assert ethanol['target'] == pytest.approx(-1.1)


def test_invalid_structure_rejected():
    rows = [{'smiles': 'not a smiles', 'measured_log_solubility': 0}]
    with pytest.raises(ValueError):
        consolidate_records(rows)
