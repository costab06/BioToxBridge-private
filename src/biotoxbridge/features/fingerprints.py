"""Morgan fingerprint featurizer (RDKit).

Generates radius-2, 2048-bit Morgan fingerprints from canonical SMILES.

Usage:
    from biotoxbridge.features.fingerprints import morgan_fp, fp_matrix

    bits = morgan_fp("CCO")          # np.ndarray shape (2048,), dtype uint8
    X    = fp_matrix(records)        # shape (n, 2048)
"""

import numpy as np


def morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Return a binary Morgan fingerprint as a uint8 numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def fp_matrix(
    records: list[dict],
    smiles_field: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute Morgan FPs for a list of structure records.

    Parameters
    ----------
    records      : enriched records from structure_groups.assign_groups()
    smiles_field : field containing canonical SMILES (default: canonical_smiles)
    radius       : Morgan radius (default: 2)
    n_bits       : bit vector length (default: 2048)

    Returns
    -------
    (X, dtxsids)
      X        : float32 ndarray shape (n, n_bits)
      dtxsids  : list of DTXSID strings, same order as rows in X
    """
    dtxsids = []
    rows = []
    for rec in records:
        smiles = str(rec.get(smiles_field) or "").strip()
        dtx    = rec.get("dtxsid", "")
        if not smiles:
            raise ValueError(f"Empty {smiles_field} for dtxsid={dtx!r}")
        rows.append(morgan_fp(smiles, radius=radius, n_bits=n_bits))
        dtxsids.append(dtx)
    X = np.array(rows, dtype=np.float32)
    return X, dtxsids
