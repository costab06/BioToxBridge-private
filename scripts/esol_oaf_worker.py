#!/usr/bin/env python3
"""OAF ESOL worker with opt-in deterministic crash after stage artifact persistence."""
import argparse
import asyncio
import os
import signal
from pathlib import Path

from biotoxbridge.oaf.agent import create_agent_class
from biotoxbridge.oaf.esol_lifecycle_orchestration import safe_lifecycle_stages


def claim_once(marker: str | Path) -> bool:
    """Persist a single-use crash marker before SIGKILL; no second crash on replay."""
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        f.write('injected\n')
        f.flush()
        os.fsync(f.fileno())
    return True


def make_worker_class(base_class, *, crash_run_id=None, crash_stage='esol_infer', crash_marker=None):
    class ESOLWorker(base_class):
        async def handle_event(self, event):
            downstream = await super().handle_event(event)
            payload = event.payload
            if (crash_run_id and isinstance(payload, dict)
                    and payload.get('run_id') == crash_run_id
                    and payload.get('stage') == crash_stage
                    and claim_once(crash_marker)):
                print('INJECTING SIGKILL AFTER ARTIFACT, BEFORE OAF COMMIT', flush=True)
                os.kill(os.getpid(), signal.SIGKILL)
            return downstream
    return ESOLWorker


async def main(args):
    from shared.eventbus.redis import RedisEventBus
    from shared.correlation.registry import CorrelationRegistry

    bus = RedisEventBus(args.redis_url)
    try:
        base = create_agent_class()
        worker = make_worker_class(
            base,
            crash_run_id=args.crash_after_artifact_run_id,
            crash_stage=args.crash_stage,
            crash_marker=args.crash_marker,
        )
        agent = worker(
            event_bus=bus,
            correlation_registry=CorrelationRegistry(bus.redis),
            artifact_root=args.artifact_root,
            stages=safe_lifecycle_stages(args.store, tox21_url=args.tox21_url),
            name=args.agent_name,
            stream=args.stream,
            min_idle_ms=args.min_idle_ms,
            heartbeat_interval=args.heartbeat_interval,
            recovery_interval=args.recovery_interval,
        )
        await agent.run()
    finally:
        await bus.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--redis-url', default='redis://localhost:6381/0')
    p.add_argument('--store', default='runs/esol-lifecycle')
    p.add_argument('--artifact-root', default='runs/oaf-artifacts')
    p.add_argument('--tox21-url', default='http://localhost:8080')
    p.add_argument('--agent-name', default='biotoxbridge-esol')
    p.add_argument('--stream', default='biotoxbridge.esol.events')
    p.add_argument('--min-idle-ms', type=int, default=30000)
    p.add_argument('--heartbeat-interval', type=float, default=10.0)
    p.add_argument('--recovery-interval', type=float, default=10.0)
    p.add_argument('--crash-after-artifact-run-id', default=None,
                   help='TEST ONLY: SIGKILL once after stage output, before OAF commit, for this run ID')
    p.add_argument('--crash-stage', default='esol_infer')
    p.add_argument('--crash-marker', default=None)
    args = p.parse_args()
    if args.crash_after_artifact_run_id:
        if not args.crash_marker:
            p.error('--crash-marker is required with --crash-after-artifact-run-id')
        if args.crash_stage == 'esol_promote':
            p.error('Do not inject SIGKILL into a model promotion stage')
    asyncio.run(main(args))
