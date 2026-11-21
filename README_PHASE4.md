# BioToxBridge Phase 4 — OAF-managed ESOL training lifecycle

This patch adds a **second OAF agent**, a completion-event coordinator, to the existing ESOL worker. The worker executes `esol_acquire`, `esol_train` (training plus frozen-holdout evaluation), and `esol_promote`. The coordinator consumes completion events and emits the next stage request in the same Redis stream, with OAF's normal atomic event commit. The existing inference stage is unchanged.

**Important:** `esol_train` already performs evaluation; this patch does not claim a separate evaluation agent. The registry's `promote` method applies the quality gate and records rejected candidates without moving the accepted pointer. A POSIX file lock serializes promotions by patched workers **on the same host/shared filesystem**. Stop older unpatched workers before using it. Do not run multiple lifecycle jobs concurrently in this validation.

## Install

```bash
cd ~/Projects/PyCharmProjects/BioToxBridge
unzip -o ~/Downloads/biotoxbridge_oaf_lifecycle_phase4_v1.zip
export PYTHONPATH="$PWD/src:$HOME/Projects/PyCharmProjects/OpenAgentFabric/framework:${PYTHONPATH:-}"
python -m pytest -q tests/test_esol_lifecycle_orchestration.py
python -m pytest -q
```

**Stop any old ESOL worker before starting the patched one.** Preserve `runs/esol-lifecycle` and `runs/oaf-artifacts`.

## Terminal A: stage worker

```bash
python scripts/esol_oaf_worker.py --redis-url redis://localhost:6381/0
```

## Terminal B: coordinator

```bash
python scripts/esol_oaf_lifecycle_coordinator.py --redis-url redis://localhost:6381/0
```

## Terminal C: acceptance-path lifecycle

Use the already downloaded, verified ESOL dataset to avoid network variability:

```bash
DATASET=runs/esol-lifecycle/datasets/8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189/source.csv
python scripts/esol_oaf_lifecycle_submit.py \
  --run-id esol-lifecycle-accept-001 \
  --source "$DATASET" \
  --expected-sha256 8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189
```

Check status and artifacts:

```bash
python scripts/esol_lifecycle.py status
find runs/oaf-artifacts/esol-lifecycle-accept-001 -maxdepth 1 -type f -printf '%f\n'
redis-cli -p 6381 XINFO GROUPS biotoxbridge.esol.events
```

## Terminal C: rejection-path lifecycle

This run deliberately sets an impossible MAE gate to force rejection, without changing the accepted model:

```bash
python scripts/esol_oaf_lifecycle_submit.py \
  --run-id esol-lifecycle-reject-001 \
  --source "$DATASET" \
  --expected-sha256 8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189 \
  --max-mae 0.000001
```

Inspect `esol_promote-*.json` under the reject run: expect `accepted: false`, with `mae_above_threshold`. Confirm `python scripts/esol_lifecycle.py status` still reports the previously accepted version, and run inference with the existing accepted model.

## Limitations

- This is an **OAF event-driven linear workflow** (acquire -> train/evaluate -> promotion), not a generic DAG scheduler.
- An interrupted training stage may create an unreferenced candidate before it creates the immutable OAF output; this is safe for the accepted pointer but not exactly-once candidate creation.
- A crash during registry promotion is **not** covered by the earlier inference SIGKILL test; history and pointer updates are not one transaction. Do not inject SIGKILL into promotion.
- `fcntl` locking only coordinates patched local processes on a filesystem with reliable advisory locks. Other clients of `ModelRegistry.promote` are not covered.
- Retraining on the identical dataset may produce a different candidate and should not be interpreted as new evidence of improved generalization. Scheduled retraining on newly labeled data remains future work.
