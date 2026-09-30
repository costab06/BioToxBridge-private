"""Structure group assignment via RDKit canonical SMILES.

Two chemicals with identical canonical SMILES receive the same structure_group
integer.  This ensures the train/val/test split never leaks stereoisomers or
salt-stripped duplicates across partitions.

Usage:
    from biotoxbridge.chemistry.structure_groups import assign_groups

    records = assign_groups(resolved)   # adds 'canonical_smiles', 'structure_group'
"""

import sys


def _canonicalize(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None if the molecule fails to parse."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def assign_groups(
    records: list[dict],
    smiles_field: str = "qsarReadySmiles",
) -> tuple[list[dict], list[dict]]:
    """
    Canonicalize SMILES and assign integer structure_group labels.

    Parameters
    ----------
    records      : list of dicts from resolver.resolve(); must contain smiles_field.
    smiles_field : which SMILES column to canonicalize (default: qsarReadySmiles).

    Returns
    -------
    (enriched, failed)
      enriched : records with 'canonical_smiles' and 'structure_group' added.
      failed   : records where RDKit could not parse the SMILES.

    Group numbering is deterministic: groups are assigned in order of first
    appearance after sorting records by dtxsid.
    """
    sorted_recs = sorted(records, key=lambda r: r.get("dtxsid", ""))

    canon_to_group: dict[str, int] = {}
    enriched: list[dict] = []
    failed: list[dict] = []

    for rec in sorted_recs:
        raw = str(rec.get(smiles_field) or "").strip()
        if not raw:
            failed.append({**rec, "failure_reason": f"{smiles_field}_empty"})
            continue
        canon = _canonicalize(raw)
        if canon is None:
            failed.append({**rec, "failure_reason": "rdkit_parse_failed"})
            continue
        if canon not in canon_to_group:
            canon_to_group[canon] = len(canon_to_group)
        enriched.append({**rec, "canonical_smiles": canon,
                         "structure_group": canon_to_group[canon]})

    n_groups = len(canon_to_group)
    print(
        f"Structure groups: {len(enriched)} records → {n_groups} unique groups "
        f"({len(failed)} failed SMILES parse)",
        file=sys.stderr,
    )
    return enriched, failed


def group_summary(enriched: list[dict]) -> dict:
    """Return a summary dict: n_chemicals, n_groups, n_singletons, n_duplicates."""
    from collections import Counter
    counts = Counter(r["structure_group"] for r in enriched)
    singletons = sum(1 for v in counts.values() if v == 1)
    return {
        "n_chemicals": len(enriched),
        "n_groups":    len(counts),
        "n_singletons": singletons,
        "n_duplicates": len(counts) - singletons,
    }
