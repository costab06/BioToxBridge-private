"""Bind existing BioToxBridge scientific stages to OAF's JSON stage contract.

Inputs are JSON mappings. Configuration and output directories are supplied by
this factory, never by untrusted event payloads. No OAF import is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from biotoxbridge import stages as scientific


def scientific_stages(config: dict, run_dir: str | Path,
                      *, interim_dir: str | Path | None = None) -> dict[str, Callable[[Any], Any]]:
    """Return allowlisted scientific stages for one isolated experiment run.

    Each stage receives one JSON-compatible mapping and returns a mapping.
    The caller must create a separate mapping for each run. The resolver cache
    defaults to the run directory; an explicit cache location may be supplied.
    """
    run_dir = Path(run_dir)
    audit = run_dir / 'audit'
    interim = Path(interim_dir) if interim_dir is not None else run_dir / 'interim'
    processed = run_dir / 'processed'
    splits = run_dir / 'splits'
    for directory in (audit, interim, processed, splits):
        directory.mkdir(parents=True, exist_ok=True)

    def mapping(value: Any, *fields: str) -> dict:
        if not isinstance(value, dict):
            raise ValueError('Scientific stage input must be a JSON object')
        for field in fields:
            if field not in value:
                raise ValueError(f'Missing scientific input: {field}')
        return value

    def target(value: Any) -> dict:
        mapping(value)
        return {'target': scientific.build_toxicity_target(config, audit)}

    def resolve(value: Any) -> dict:
        value = mapping(value, 'target')
        resolved, missing = scientific.resolve_chemistry(value['target'], interim, audit)
        return {'target': value['target'], 'resolved': resolved, 'missing': missing}

    def group(value: Any) -> dict:
        value = mapping(value, 'target', 'resolved')
        enriched, failed = scientific.group_structures(value['resolved'], config)
        return {**value, 'enriched': enriched, 'group_failed': failed}

    def score(value: Any) -> dict:
        value = mapping(value, 'target', 'enriched')
        scored = scientific.score_tox21(value['enriched'], config, audit)
        return {**value, 'scored': scored}

    def master(value: Any) -> dict:
        value = mapping(value, 'target', 'scored')
        records, excluded = scientific.assemble_master(
            value['target'], value['scored'], processed, audit)
        if not records:
            raise ValueError('Empty master dataset — check audit logs')
        return {'master': records, 'excluded': excluded}

    def split(value: Any) -> dict:
        value = mapping(value, 'master')
        train, test = scientific.freeze_split(value['master'], config, splits)
        return {'train': train, 'test': test, 'excluded': value.get('excluded', [])}

    return {
        'toxicity_target': target,
        'chemistry_resolution': resolve,
        'structure_grouping': group,
        'tox21_scoring': score,
        'master_assembly': master,
        'dataset_split': split,
    }
