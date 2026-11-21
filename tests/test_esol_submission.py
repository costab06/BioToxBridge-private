
"""Redis-backed tests for atomic ESOL lifecycle submission.

Run:
    python -m pytest -v tests/test_esol_submission.py

Requires:
    Redis at redis://localhost:6381/0
    OpenAgentFabric/framework and BioToxBridge/src on PYTHONPATH.

Tests use unique Redis keys and clean up after themselves.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from shared.correlation.registry import CorrelationRecord
from shared.eventbus.events import EventEnvelope

from biotoxbridge.oaf.submission import (
    fingerprint,
    submit_once,
)


REDIS_URL = "redis://localhost:6381/0"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def redis_client():
    client = Redis.from_url(
        REDIS_URL,
        decode_responses=False,
    )

    try:
        await client.ping()
        yield client
    finally:
        await client.aclose()


def make_submission(run_id: str):
    """Construct deterministic event and correlation identities."""

    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)

    correlation_id = uuid5(
        NAMESPACE_URL,
        f"test:correlation:{run_id}",
    )

    correlation = CorrelationRecord(
        correlation_id=correlation_id,
        trigger_type="test.esol.lifecycle",
        trigger_timestamp=timestamp,
        business_fields={"run_id": run_id},
        created_at=timestamp,
    )

    event = EventEnvelope(
        event_id=uuid5(
            NAMESPACE_URL,
            f"test:event:{run_id}",
        ),
        event_type="biotoxbridge.stage.requested",
        source_agent="test-submit",
        correlation_id=correlation_id,
        payload={
            "run_id": run_id,
            "stage": "esol_acquire",
        },
    )

    return event, correlation


def submission_key(run_id: str) -> str:
    return f"biotoxbridge:submission:{run_id}"


def correlation_key(correlation: CorrelationRecord) -> str:
    return f"oaf:correlation:{correlation.correlation_id}"


async def cleanup(redis, run_id, stream, correlation):
    await redis.delete(
        submission_key(run_id),
        correlation_key(correlation),
        stream,
    )


async def submit(
    redis,
    run_id,
    stream,
    event,
    correlation,
    fingerprint_value="same-input",
):
    return await submit_once(
        redis,
        stream=stream,
        run_id=run_id,
        fingerprint_value=fingerprint_value,
        event=event,
        correlation_record=correlation,
    )


@pytest.mark.anyio
async def test_first_submission_creates_one_event(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        result = await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
        )

        assert result.created is True
        assert result.correlation_id == correlation.correlation_id
        assert result.message_id

        assert await redis_client.xlen(stream) == 1

        messages = await redis_client.xrange(stream)

        assert len(messages) == 1

        message_id, fields = messages[0]

        assert message_id.decode() == result.message_id

        published = EventEnvelope.model_validate_json(
            fields[b"event"]
        )

        assert published.event_id == event.event_id
        assert published.correlation_id == event.correlation_id
        assert published.payload == event.payload

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_identical_retry_does_not_republish(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        first = await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
        )

        second = await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
        )

        assert first.created is True
        assert second.created is False

        assert second.message_id == first.message_id
        assert second.correlation_id == first.correlation_id

        assert await redis_client.xlen(stream) == 1

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_concurrent_submission_is_idempotent(redis_client):
    """Twenty concurrent attempts must publish exactly once."""

    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        async def attempt():
            return await submit(
                redis_client,
                run_id,
                stream,
                event,
                correlation,
            )

        results = await asyncio.gather(
            *(attempt() for _ in range(20))
        )

        assert sum(result.created for result in results) == 1

        assert len({
            result.message_id for result in results
        }) == 1

        assert len({
            result.correlation_id for result in results
        }) == 1

        assert await redis_client.xlen(stream) == 1

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_changed_fingerprint_is_rejected(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
            fingerprint_value="original-input",
        )

        with pytest.raises(
            ResponseError,
            match="SUBMISSION_CONFLICT",
        ):
            await submit(
                redis_client,
                run_id,
                stream,
                event,
                correlation,
                fingerprint_value="changed-input",
            )

        assert await redis_client.xlen(stream) == 1

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_correlation_record_is_persistent(redis_client):
    """Submission correlation must not expire after 24 hours."""

    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
        )

        key = correlation_key(correlation)

        stored = await redis_client.get(key)

        assert stored is not None

        restored = CorrelationRecord.model_validate_json(stored)

        assert restored.correlation_id == correlation.correlation_id
        assert restored.business_fields == correlation.business_fields

        # -1 means the key exists without an expiration.
        assert await redis_client.ttl(key) == -1

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_retry_restores_missing_correlation(redis_client):
    """An existing submission can repair missing correlation metadata."""

    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        first = await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
        )

        key = correlation_key(correlation)

        await redis_client.delete(key)

        assert await redis_client.exists(key) == 0

        second = await submit(
            redis_client,
            run_id,
            stream,
            event,
            correlation,
        )

        assert second.created is False
        assert second.message_id == first.message_id

        assert await redis_client.exists(key) == 1
        assert await redis_client.ttl(key) == -1

        assert await redis_client.xlen(stream) == 1

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_conflicting_correlation_is_rejected(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        key = correlation_key(correlation)

        await redis_client.set(
            key,
            '{"correlation_id":"conflicting-record"}',
        )

        with pytest.raises(
            ResponseError,
            match="CORRELATION_CONFLICT",
        ):
            await submit(
                redis_client,
                run_id,
                stream,
                event,
                correlation,
            )

        assert await redis_client.xlen(stream) == 0
        assert await redis_client.exists(submission_key(run_id)) == 0

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


@pytest.mark.anyio
async def test_event_correlation_identity_must_match(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    different_run = f"test-{uuid4()}"
    _, different_correlation = make_submission(different_run)

    try:
        with pytest.raises(
            ValueError,
            match="identities differ",
        ):
            await submit(
                redis_client,
                run_id,
                stream,
                event,
                different_correlation,
            )

        assert await redis_client.xlen(stream) == 0

    finally:
        await cleanup(
            redis_client, run_id, stream, correlation
        )


def test_fingerprint_is_order_independent():
    a = {
        "stream": "biotoxbridge.esol.events",
        "config": {
            "seed": 42,
            "policy": {
                "max_mae": 1.0,
                "min_r2": 0.2,
            },
        },
    }

    b = {
        "config": {
            "policy": {
                "min_r2": 0.2,
                "max_mae": 1.0,
            },
            "seed": 42,
        },
        "stream": "biotoxbridge.esol.events",
    }

    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_detects_changed_configuration():
    original = {
        "seed": 42,
        "max_mae": 1.0,
    }

    changed = {
        "seed": 43,
        "max_mae": 1.0,
    }

    assert fingerprint(original) != fingerprint(changed)

@pytest.mark.anyio
async def test_invalid_stream_type_has_no_side_effects(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        await redis_client.set(stream, "not-a-stream")

        with pytest.raises(ResponseError, match="INVALID_STREAM_KEY"):
            await submit(
                redis_client, run_id, stream, event, correlation
            )

        assert await redis_client.exists(submission_key(run_id)) == 0
        assert await redis_client.exists(correlation_key(correlation)) == 0
        assert await redis_client.get(stream) == b"not-a-stream"
    finally:
        await cleanup(redis_client, run_id, stream, correlation)


@pytest.mark.anyio
async def test_invalid_correlation_type_has_no_side_effects(redis_client):
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        await redis_client.lpush(correlation_key(correlation), "invalid")

        with pytest.raises(ResponseError, match="INVALID_CORRELATION_KEY"):
            await submit(
                redis_client, run_id, stream, event, correlation
            )

        assert await redis_client.xlen(stream) == 0
        assert await redis_client.exists(submission_key(run_id)) == 0
    finally:
        await cleanup(redis_client, run_id, stream, correlation)


@pytest.mark.anyio
async def test_retry_after_lost_response_does_not_republish(redis_client):
    """Simulate a successful commit whose result the caller never receives."""
    run_id = f"test-{uuid4()}"
    stream = f"test:esol:submission:{uuid4()}"
    event, correlation = make_submission(run_id)

    try:
        # Redis commits the submission, but the caller discards the result.
        await submit(redis_client, run_id, stream, event, correlation)

        # A new attempt must recover the original committed identity.
        retry = await submit(
            redis_client, run_id, stream, event, correlation
        )

        messages = await redis_client.xrange(stream)

        assert retry.created is False
        assert len(messages) == 1
        assert retry.message_id == messages[0][0].decode()
        assert await redis_client.xlen(stream) == 1
    finally:
        await cleanup(redis_client, run_id, stream, correlation)
