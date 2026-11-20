#!/usr/bin/env python3
"""Submit acquire -> train/evaluate -> conditional promotion to OAF."""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5
from biotoxbridge.oaf.agent import EVENT_REQUESTED
from biotoxbridge.oaf.artifacts import publish_json
from biotoxbridge.oaf.esol_lifecycle_orchestration import STREAM

async def main(a):
    from shared.eventbus.redis import RedisEventBus
    from shared.eventbus.events import EventEnvelope
    from shared.correlation.registry import CorrelationRegistry
    config = {'lifecycle_v1': True, 'seed': a.seed,
              'policy': {'max_mae': a.max_mae, 'min_r2': a.min_r2,
                         'max_mae_regression': a.max_mae_regression}}
    publish_json(a.artifact_root, a.run_id, 'lifecycle-config', config)
    ref = publish_json(a.artifact_root, a.run_id, 'lifecycle-esol_acquire-input',
                       {'source': a.source, **({'expected_sha256': a.expected_sha256} if a.expected_sha256 else {})})
    bus = RedisEventBus(a.redis_url)
    try:
        correlation = await CorrelationRegistry(bus.redis).create(
            'biotoxbridge.esol.lifecycle', datetime.now(timezone.utc), {'run_id': a.run_id})
        event = EventEnvelope(event_id=uuid5(NAMESPACE_URL, f'biotoxbridge:lifecycle:{a.run_id}:esol_acquire'),
                              event_type=EVENT_REQUESTED, source_agent='biotoxbridge-lifecycle-submit',
                              correlation_id=correlation.correlation_id,
                              payload={'run_id': a.run_id, 'stage': 'esol_acquire', 'input_artifact': ref})
        message_id = await bus.publish(a.stream, event)
        print(json.dumps({'run_id': a.run_id, 'message_id': message_id,
                          'correlation_id': str(correlation.correlation_id)}, indent=2))
    finally:
        await bus.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--redis-url', default='redis://localhost:6381/0')
    p.add_argument('--stream', default=STREAM)
    p.add_argument('--artifact-root', default='runs/oaf-artifacts')
    p.add_argument('--run-id', required=True)
    p.add_argument('--source', default='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv')
    p.add_argument('--expected-sha256')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-mae', type=float, default=1.0)
    p.add_argument('--min-r2', type=float, default=0.2)
    p.add_argument('--max-mae-regression', type=float, default=0.0)
    asyncio.run(main(p.parse_args()))
