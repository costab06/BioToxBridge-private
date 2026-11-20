# ESOL Redis stream isolation fix

Patch the two scripts in the BioToxBridge repository. This keeps the old Redis stream intact and routes ESOL to `biotoxbridge.esol.events`.

From BioToxBridge project root:

```bash
unzip -o ~/Downloads/biotoxbridge_esol_stream_isolation_fix_v1.zip
export PYTHONPATH="$PWD/src:$HOME/Projects/PyCharmProjects/OpenAgentFabric/framework:${PYTHONPATH:-}"
python scripts/esol_oaf_worker.py --redis-url redis://localhost:6381/0
```

In another terminal with the same PYTHONPATH:

```bash
python scripts/esol_oaf_submit.py --redis-url redis://localhost:6381/0 \
  --run-id esol-oaf-002 --request-id ethanol-002 --stage esol_infer \
  --request '{"smiles":"CCO","model_version":"6fb08776038c4224817833d8a3ad404a","include_tox21":true}'
```

The old `oaf.events` stream and its pending messages are deliberately untouched. The existing `biotoxbridge-esol` group on the old stream may retain pending failures; it is no longer consumed by this worker.
