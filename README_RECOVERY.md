# ESOL OAF deterministic SIGKILL recovery test

**TEST ONLY.** No OpenAgentFabric files are modified. Replace only `scripts/esol_oaf_worker.py`.

1. Stop the existing ESOL worker with Ctrl+C. Ensure no second worker is running in the same group.
2. Install this ZIP from the BioToxBridge repository root.
3. Use a fresh run ID and unique crash-marker filename. Never delete the marker between crash and recovery.
4. Start the worker with the injection flags (below). The process is expected to be killed with SIGKILL.
5. Submit a new inference request from a second terminal.
6. Verify the stage artifact exists and Redis group pending count is 1.
7. Restart the worker WITHOUT injection flags. Wait for recovery, then verify pending is 0, lag is 0 and only one completion event exists for the request.

## Terminal A: injected worker

```bash
export PYTHONPATH="$PWD/src:$HOME/Projects/PyCharmProjects/OpenAgentFabric/framework:${PYTHONPATH:-}"
python scripts/esol_oaf_worker.py \
  --redis-url redis://localhost:6381/0 \
  --min-idle-ms 3000 --heartbeat-interval 1 --recovery-interval 1 \
  --crash-after-artifact-run-id esol-kill-001 \
  --crash-marker runs/oaf-fault-markers/esol-kill-001.marker
```

## Terminal B: submit (same PYTHONPATH)

```bash
python scripts/esol_oaf_submit.py \
  --run-id esol-kill-001 --request-id ethanol-kill-001 \
  --stage esol_infer \
  --request '{"smiles":"CCO","model_version":"6fb08776038c4224817833d8a3ad404a","include_tox21":true}'
```

## Inspect before recovery

```bash
redis-cli -p 6381 XINFO GROUPS biotoxbridge.esol.events
find runs/oaf-artifacts/esol-kill-001 -type f -printf '%f\n'
cat runs/oaf-fault-markers/esol-kill-001.marker
```

## Terminal A: recovery worker (NO crash flags)

```bash
python scripts/esol_oaf_worker.py \
  --redis-url redis://localhost:6381/0 \
  --min-idle-ms 3000 --heartbeat-interval 1 --recovery-interval 1
```

After several seconds:

```bash
redis-cli -p 6381 XINFO GROUPS biotoxbridge.esol.events
redis-cli -p 6381 XRANGE biotoxbridge.esol.events - +
```

Check the completion event's `causation_id` equals the submitted event ID, and that exactly one such completion exists. A second computation can be attempted on redelivery; the existing immutable artifact must be identical, and OAF's atomic commit should emit exactly one completion event. This test does **not** prove exactly-once execution of external side effects.
