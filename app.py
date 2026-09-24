#!/usr/bin/env python3
"""BioToxBridge — multi-dataset toxicity prediction platform.

Extends Tox21 screening with additional toxicological endpoints by
aggregating multiple named models served through nnsd_client.

Architecture:
  - Tox21 assay models (classification, Morgan fingerprint input)
  - Additional endpoints (regression or classification, Morgan FP or tabular)
  - All inference routed through nnsd_client's /api/models/* API
  - SMILES conversion handled here via RDKit; nnsd_client stays data-agnostic
"""

import csv
import json
import os
import random

import requests
from flask import Flask, jsonify, render_template, request

# ── Configuration ──────────────────────────────────────────────────────────────

NNSD_URL    = os.environ.get("NNSD_URL",    "http://localhost:8080")
NNSD_KEY    = os.environ.get("NNSD_KEY",    "admin-secret-key-change-me")
TOX21_URL   = os.environ.get("TOX21_URL",   "http://localhost:8081")  # optional: pull tox21 predictions
DATA_ROOT   = os.environ.get("BTB_DATA",    "/opt/neural/data/biotoxbridge")
TOX21_DATA  = os.environ.get("TOX21_DATA",  "/opt/neural/data/tox21")
HTTP_PORT   = int(os.environ.get("BTB_PORT", "8082"))

FP_RADIUS = 2
FP_BITS   = 2048
TRAIN_FRAC = 0.80
RAND_SEED  = 42

# ── Model registry ─────────────────────────────────────────────────────────────
# Each entry describes one named model registered in nnsd_client.
#
# Fields:
#   model      - nnsd_client model name (used in /api/models/<name>/*)
#   type       - "classification" | "regression"
#   input      - "smiles"   → convert SMILES to Morgan fingerprint before predict
#                "tabular"  → send feature rows directly (caller supplies features)
#   group      - display grouping in the UI
#   label      - human-readable name
#   units      - regression output units (ignored for classification)
#   description - tooltip / context text
#   data_source - which data directory holds the raw files
#   train_file  - nnsd-relative path to the training CSV (used by /api/models/<name>/train)
#   train_params - hyperparameter overrides sent to nnsd_client at train time

MODELS = {
    # ── Tox21 nuclear receptor assays ──────────────────────────────────────────
    "NR-AR": {
        "model": "nr_ar", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "Androgen Receptor",
        "description": "Endocrine disruption; flagged in reproductive toxicity screens.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_ar_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "NR-AR-LBD": {
        "model": "nr_ar_lbd", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "Androgen Receptor LBD",
        "description": "Selectivity assay complementing full-length AR screen.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_ar_lbd_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "NR-AhR": {
        "model": "nr_ahr", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "Aryl Hydrocarbon Receptor",
        "description": "Predicts dioxin-like toxicity and CYP1A1/1A2 induction.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_ahr_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "NR-Aromatase": {
        "model": "nr_aromatase", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "Aromatase (CYP19A1)",
        "description": "Disrupts estrogen biosynthesis; reproductive and endocrine toxicity.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_aromatase_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "NR-ER": {
        "model": "nr_er", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "Estrogen Receptor α",
        "description": "Key endocrine disruptor target; linked to breast cancer risk.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_er_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "NR-ER-LBD": {
        "model": "nr_er_lbd", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "Estrogen Receptor LBD",
        "description": "Selectivity assay for ERα LBD interactions.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_er_lbd_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "NR-PPAR-gamma": {
        "model": "nr_ppar_gamma", "type": "classification", "input": "smiles",
        "group": "Tox21 — Nuclear Receptor",
        "label": "PPARγ",
        "description": "Metabolic / adipogenic effects; relevant in diabetes and obesity research.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/nr_ppar_gamma_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    # ── Tox21 stress response assays ───────────────────────────────────────────
    "SR-ARE": {
        "model": "sr_are", "type": "classification", "input": "smiles",
        "group": "Tox21 — Stress Response",
        "label": "Antioxidant Response Element",
        "description": "Oxidative stress sentinel; flags electrophilic and ROS-generating compounds.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/sr_are_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "SR-ATAD5": {
        "model": "sr_atad5", "type": "classification", "input": "smiles",
        "group": "Tox21 — Stress Response",
        "label": "ATAD5 Genotoxicity",
        "description": "DNA-damage indicator; positives often require genotoxicity follow-up.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/sr_atad5_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "SR-HSE": {
        "model": "sr_hse", "type": "classification", "input": "smiles",
        "group": "Tox21 — Stress Response",
        "label": "Heat Shock Element",
        "description": "Proteotoxic stress marker; identifies compounds that misfold proteins.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/sr_hse_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "SR-MMP": {
        "model": "sr_mmp", "type": "classification", "input": "smiles",
        "group": "Tox21 — Stress Response",
        "label": "Mitochondrial Membrane Potential",
        "description": "Primary mitochondrial toxicity screen; precursor to cytotoxicity.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/sr_mmp_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    "SR-p53": {
        "model": "sr_p53", "type": "classification", "input": "smiles",
        "group": "Tox21 — Stress Response",
        "label": "p53 Pathway",
        "description": "Genotoxicity / DNA damage; mechanistically related to SR-ATAD5.",
        "data_source": TOX21_DATA,
        "train_file": "/tox21/sr_p53_train.csv",
        "train_params": {"categorical": True, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 95.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.5, "timeout": 1200},
    },
    # ── Additional endpoints ───────────────────────────────────────────────────
    "Liver-LOAEL": {
        "model": "liver_tox", "type": "regression", "input": "smiles",
        "group": "Additional Endpoints",
        "label": "Liver LOAEL",
        "units": "log₁₀(mg/kg/day)",
        "description": "Predicts lowest-observed-adverse-effect level for liver toxicity "
                       "from ToxRefDB animal studies.",
        "data_source": DATA_ROOT,
        "train_file": "/tox21/liver_tox21_train.csv",
        "train_params": {"categorical": False, "groundTruthFirst": True,
                         "trainingSuccessPercentage": 97.0, "weightDecay": 1e-4,
                         "dropoutRate": 0.0, "timeout": 1200, "useAdam": True},
    },
    # ── Future models: add entries here ───────────────────────────────────────
    # "Kidney-LOAEL": { ... },
    # "hERG": { ... },
}


# ── RDKit ──────────────────────────────────────────────────────────────────────

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False


# ── Shared helpers ─────────────────────────────────────────────────────────────

def nnsd_headers():
    h = {"Content-Type": "application/json"}
    if NNSD_KEY:
        h["Authorization"] = f"Bearer {NNSD_KEY}"
    return h


def smiles_to_fp(smiles: str):
    """Return Morgan fingerprint as a list of floats, or None if SMILES is invalid."""
    if not RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return _morgan_gen.GetFingerprintAsNumPy(mol).tolist()


def _nnsd_model_map() -> dict:
    """Fetch nnsd_client model registry; returns {name: record} on success."""
    try:
        r = requests.get(f"{NNSD_URL}/api/models", headers=nnsd_headers(), timeout=5)
        if r.ok:
            return {m["name"]: m for m in r.json()}
    except Exception:
        pass
    return {}


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# ── Model status ───────────────────────────────────────────────────────────────

@app.route("/api/models")
def get_models():
    """Return all configured models with their training status from nnsd_client."""
    model_map = _nnsd_model_map()
    result = []
    for key, meta in MODELS.items():
        m = model_map.get(meta["model"], {})
        result.append({
            "key":         key,
            "model":       meta["model"],
            "type":        meta["type"],
            "input":       meta["input"],
            "group":       meta["group"],
            "label":       meta["label"],
            "units":       meta.get("units", ""),
            "description": meta["description"],
            "trained":     m.get("trained", False),
            "accuracy":    m.get("lastAccuracy"),
            "trainDate":   m.get("lastTrainDate"),
        })
    return jsonify(result)


# ── Training ───────────────────────────────────────────────────────────────────

@app.route("/api/models/<key>/train", methods=["POST"])
def train_model(key):
    """Submit a training job for the named model to nnsd_client."""
    if key not in MODELS:
        return jsonify({"error": f"Unknown model: {key}"}), 400

    meta   = MODELS[key]
    body   = request.get_json() or {}
    retrain = body.get("retrain", False)

    payload = dict(meta["train_params"])
    payload["infile"]  = meta["train_file"]
    payload["retrain"] = retrain

    try:
        r = requests.post(
            f"{NNSD_URL}/api/models/{meta['model']}/train",
            json=payload, headers=nnsd_headers(), timeout=10,
        )
        return jsonify(r.json()), r.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


# ── Prediction ─────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    """Run one or more SMILES through all trained smiles-input models.

    Request body:
      {
        "smiles":  ["CCO", "c1ccccc1", ...],
        "models":  ["NR-AhR", "Liver-LOAEL", ...]   // optional filter; default = all
      }

    Response:
      {
        "compounds": [
          {
            "smiles": "CCO",
            "valid":  true,
            "predictions": {
              "NR-AhR":     {"class": 0, "score": 0.12},
              "Liver-LOAEL": {"value": 2.35, "units": "log₁₀(mg/kg/day)"}
            }
          }, ...
        ]
      }
    """
    if not RDKIT_OK:
        return jsonify({"error": "RDKit not available — cannot convert SMILES"}), 500

    body         = request.get_json() or {}
    smiles_list  = body.get("smiles", [])
    model_filter = set(body.get("models", list(MODELS.keys())))

    if not smiles_list:
        return jsonify({"error": "smiles list required"}), 400

    # Convert SMILES → fingerprints
    compounds = []
    for smi in smiles_list:
        fp = smiles_to_fp(smi)
        compounds.append({"smiles": smi, "valid": fp is not None,
                           "fp": fp, "predictions": {}})

    valid = [c for c in compounds if c["valid"]]
    if not valid:
        for c in compounds:
            del c["fp"]
        return jsonify({"compounds": compounds})

    rows = [c["fp"] for c in valid]

    # Run each model
    for key, meta in MODELS.items():
        if key not in model_filter:
            continue
        if meta["input"] != "smiles":
            continue

        model_name = meta["model"]
        try:
            r = requests.post(
                f"{NNSD_URL}/api/models/{model_name}/predict",
                json={"rows": rows},
                headers=nnsd_headers(),
                timeout=30,
            )
            if not r.ok:
                continue
            preds = r.json().get("predictions", [])
            for i, c in enumerate(valid):
                if i >= len(preds):
                    break
                p = preds[i]
                if meta["type"] == "regression":
                    c["predictions"][key] = {
                        "value": p.get("value"),
                        "units": meta.get("units", ""),
                    }
                else:
                    c["predictions"][key] = {
                        "class": p.get("class"),
                        "score": p.get("score"),
                    }
        except requests.RequestException:
            pass  # model not trained or unreachable; silently omit

    for c in compounds:
        del c["fp"]

    return jsonify({"compounds": compounds})


@app.route("/api/predict/tabular", methods=["POST"])
def predict_tabular():
    """Run pre-computed feature rows through a single tabular-input model.

    Request body:
      {
        "model": "Liver-LOAEL",
        "rows":  [[f0, f1, ...], ...]
      }
    """
    body = request.get_json() or {}
    key  = body.get("model")
    rows = body.get("rows", [])

    if not key or key not in MODELS:
        return jsonify({"error": "model key required and must be in MODELS"}), 400
    if not rows:
        return jsonify({"error": "rows required"}), 400

    meta = MODELS[key]
    if meta["input"] != "tabular":
        return jsonify({"error": f"{key} expects SMILES input, not tabular"}), 400

    try:
        r = requests.post(
            f"{NNSD_URL}/api/models/{meta['model']}/predict",
            json={"rows": rows},
            headers=nnsd_headers(),
            timeout=30,
        )
        return jsonify(r.json()), r.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


# ── Tox21 bridge ───────────────────────────────────────────────────────────────

@app.route("/api/import/tox21", methods=["POST"])
def import_tox21():
    """Pull predictions from the running tox21_demo and return them combined
    with BioToxBridge model predictions for the same SMILES.

    Request body:
      { "smiles": ["CCO", ...] }

    Calls tox21_demo /api/predict, then runs the same SMILES through all
    additional (non-Tox21) BioToxBridge models and merges the results.
    """
    body        = request.get_json() or {}
    smiles_list = body.get("smiles", [])
    if not smiles_list:
        return jsonify({"error": "smiles list required"}), 400

    # Fetch Tox21 predictions
    tox21_preds = {}
    try:
        r = requests.post(
            f"{TOX21_URL}/api/predict",
            json={"smiles": smiles_list},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if r.ok:
            for c in r.json().get("compounds", []):
                tox21_preds[c["smiles"]] = c.get("predictions", {})
    except requests.RequestException:
        pass  # tox21_demo unavailable; continue with BioToxBridge models only

    # Additional BioToxBridge models (non-Tox21 groups)
    extra_keys = [k for k, m in MODELS.items() if not m["group"].startswith("Tox21")]
    btb_resp = {}
    if extra_keys and RDKIT_OK:
        try:
            pr = requests.post(
                "http://localhost:{}/api/predict".format(HTTP_PORT),
                json={"smiles": smiles_list, "models": extra_keys},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if pr.ok:
                for c in pr.json().get("compounds", []):
                    btb_resp[c["smiles"]] = c.get("predictions", {})
        except requests.RequestException:
            pass

    # Merge
    result = []
    for smi in smiles_list:
        merged = {}
        merged.update(tox21_preds.get(smi, {}))
        merged.update(btb_resp.get(smi, {}))
        result.append({"smiles": smi, "predictions": merged})

    return jsonify({"compounds": result})


# ── Model history ──────────────────────────────────────────────────────────────

@app.route("/api/models/<key>/history")
def model_history(key):
    if key not in MODELS:
        return jsonify({"error": f"Unknown model: {key}"}), 400
    meta = MODELS[key]
    try:
        r = requests.get(f"{NNSD_URL}/api/models/{meta['model']}/history",
                         headers=nnsd_headers(), timeout=5)
        return jsonify(r.json()), r.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=HTTP_PORT, debug=False)
