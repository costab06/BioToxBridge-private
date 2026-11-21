"""Run BioToxBridge scientific preparation through OAF Redis Streams.

Usage (with OpenAgentFabric/framework on PYTHONPATH):
 python -m biotoxbridge.oaf.runtime --config config/experiments/liver_loael_v1.yaml --run-id demo-1

This is a development integration runner, not a production exactly-once
execution guarantee for external scoring or non-transactional audit files.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import yaml

from biotoxbridge.oaf.coordinator import EVENT_STARTED, EVENT_FINISHED, create_coordinator_class
from biotoxbridge.oaf.agent import create_agent_class
from biotoxbridge.oaf.scientific import scientific_stages


def load_configuration(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def run_paths(run_id: str, artifact_root: str | Path, work_root: str | Path):
    # Reuse the artifact contract's path validation.
    from biotoxbridge.oaf.artifacts import _safe_path
    _safe_path(Path(artifact_root), run_id, "validation")
    return Path(artifact_root).resolve(), Path(work_root).resolve() / run_id


def config_fingerprint(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


async def run_workflow(*, config: dict, run_id: str, artifact_root: str | Path,
                       work_root: str | Path, redis_url: str,
                       timeout_seconds: float = 900, start: bool = True):
    from shared.correlation.registry import CorrelationRegistry
    from shared.eventbus.events import EventEnvelope
    from shared.eventbus.redis import RedisEventBus

    artifact_root, work_dir = run_paths(run_id, artifact_root, work_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the configuration for this run; reject changes on restart.
    config_path = work_dir / "oaf_config.json"
    encoded = json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with config_path.open("x") as f:
            f.write(encoded)
    except FileExistsError:
        if config_path.read_text() != encoded:
            raise ValueError("Run configuration changed; choose a new run_id")

    bus = RedisEventBus(redis_url)
    registry = CorrelationRegistry(bus.redis)
    coordinator = create_coordinator_class()(
        event_bus=bus, correlation_registry=registry, artifact_root=artifact_root
    )
    agent = create_agent_class()(
        event_bus=bus, correlation_registry=registry, artifact_root=artifact_root,
        stages=scientific_stages(config, work_dir),
    )

    # Stable correlation identity across restarts; an epoch timestamp is used
    # solely as a deterministic identity input, not as an event occurrence time.
    identity_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    record = await registry.create(EVENT_STARTED, identity_time, {
        "run_id": run_id, "config_sha256": config_fingerprint(config)
    })
    completion_key = f"oaf:workflow:completion:{record.correlation_id}:{EVENT_FINISHED}"
    tasks = []
    try:
        await bus.redis.ping()
        if start:
            # Only one bootstrap publish for a given correlation. Atomic SET NX
            # avoids duplicate starts, but cannot atomically couple with XADD.
            # If bootstrap crashes after SET and before XADD, manual recovery is needed.
            bootstrap_key = f"biotoxbridge:bootstrap:{record.correlation_id}"
            first = await bus.redis.set(bootstrap_key, "claimed", nx=True)
            if first:
                event = EventEnvelope(
                    event_id=uuid5(NAMESPACE_URL, str(record.correlation_id) + ":start"),
                    event_type=EVENT_STARTED, source_agent="biotoxbridge-bootstrap",
                    correlation_id=record.correlation_id,
                    payload={"run_id": run_id},
                )
                await bus.publish("oaf.events", event)
                await bus.redis.set(bootstrap_key, "published")
        tasks = [asyncio.create_task(coordinator.run()), asyncio.create_task(agent.run())]
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            value = await bus.redis.get(completion_key)
            if value:
                finished = EventEnvelope.model_validate_json(value)
                return finished.payload
            for task in tasks:
                if task.done():
                    task.result()  # surface unexpected worker termination
            await asyncio.sleep(0.5)
        raise TimeoutError(f"No workflow completion after {timeout_seconds} seconds; inspect Redis DLQ")
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await bus.redis.aclose()


def main():
    parser = argparse.ArgumentParser(description="BioToxBridge OAF Redis integration runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="runs/oaf-artifacts")
    parser.add_argument("--work-root", default="runs/oaf-work")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true", help="Do not publish a new start event")
    args = parser.parse_args()
    result = asyncio.run(run_workflow(
        config=load_configuration(args.config), run_id=args.run_id,
        artifact_root=args.artifact_root, work_root=args.work_root,
        redis_url=args.redis_url, timeout_seconds=args.timeout,
        start=not args.resume,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
