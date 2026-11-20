#!/usr/bin/env python3
"""Run an independent OAF consumer for lifecycle completion events."""
import argparse
import asyncio
from biotoxbridge.oaf.esol_lifecycle_orchestration import STREAM, create_coordinator_class

async def main(a):
    from shared.eventbus.redis import RedisEventBus
    from shared.correlation.registry import CorrelationRegistry
    bus = RedisEventBus(a.redis_url)
    try:
        agent = create_coordinator_class()(
            event_bus=bus, correlation_registry=CorrelationRegistry(bus.redis),
            artifact_root=a.artifact_root, stream=a.stream,
            min_idle_ms=a.min_idle_ms, heartbeat_interval=a.heartbeat_interval,
            recovery_interval=a.recovery_interval)
        await agent.run()
    finally:
        await bus.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--redis-url', default='redis://localhost:6381/0')
    p.add_argument('--stream', default=STREAM)
    p.add_argument('--artifact-root', default='runs/oaf-artifacts')
    p.add_argument('--min-idle-ms', type=int, default=30000)
    p.add_argument('--heartbeat-interval', type=float, default=10)
    p.add_argument('--recovery-interval', type=float, default=10)
    asyncio.run(main(p.parse_args()))
