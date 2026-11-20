#!/usr/bin/env python3
"""Submit one durable BioToxBridge stage request to OAF Redis Streams."""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_URL
from biotoxbridge.oaf.agent import EVENT_REQUESTED
from biotoxbridge.oaf.artifacts import publish_json

async def main(a):
    from shared.eventbus.redis import RedisEventBus
    from shared.eventbus.events import EventEnvelope
    from shared.correlation.registry import CorrelationRegistry
    bus = RedisEventBus(a.redis_url)
    try:
        request = json.loads(a.request)
        ref = publish_json(a.artifact_root, a.run_id, a.request_id, request)
        # Stable event ID avoids accidental identity changes across retries.
        eid = uuid5(NAMESPACE_URL, f'biotoxbridge:{a.run_id}:{a.request_id}:{a.stage}')
        registry = CorrelationRegistry(bus.redis)
        corr = await registry.create('biotoxbridge.esol', datetime.now(timezone.utc),
                                     {'run_id': a.run_id, 'request_id': a.request_id})
        event = EventEnvelope(event_id=eid, event_type=EVENT_REQUESTED,
                              source_agent='biotoxbridge-submit', correlation_id=corr.correlation_id,
                              payload={'run_id': a.run_id, 'stage': a.stage,
                                       'input_artifact': ref})
        mid = await bus.publish(a.stream, event)
        print(json.dumps({'message_id': mid, 'event_id': str(eid),
                          'correlation_id': str(corr.correlation_id), 'input_artifact': ref}, indent=2))
    finally:
        await bus.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--redis-url', default='redis://localhost:6381/0')
    p.add_argument('--artifact-root', default='runs/oaf-artifacts')
    p.add_argument('--stream', default='biotoxbridge.esol.events')
    p.add_argument('--run-id', required=True)
    p.add_argument('--request-id', required=True)
    p.add_argument('--stage', required=True, choices=['esol_acquire','esol_train','esol_promote','esol_infer'])
    p.add_argument('--request', required=True, help='JSON object for the stage')
    asyncio.run(main(p.parse_args()))
