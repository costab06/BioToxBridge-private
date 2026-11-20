
"""Atomic, idempotent ESOL workflow submission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from biotoxbridge.oaf.artifacts import _safe_path


SUBMIT_SCRIPT = """
local submission_type = redis.call('TYPE', KEYS[1]).ok
local stream_type = redis.call('TYPE', KEYS[2]).ok
local correlation_type = redis.call('TYPE', KEYS[3]).ok

if submission_type ~= 'none' and submission_type ~= 'string' then
    return redis.error_reply('INVALID_SUBMISSION_KEY')
end

if stream_type ~= 'none' and stream_type ~= 'stream' then
    return redis.error_reply('INVALID_STREAM_KEY')
end

if correlation_type ~= 'none' and correlation_type ~= 'string' then
    return redis.error_reply('INVALID_CORRELATION_KEY')
end

local existing = redis.call('GET', KEYS[1])
local correlation = redis.call('GET', KEYS[3])

if existing then
    local record = cjson.decode(existing)

    if record.fingerprint ~= ARGV[1] or
       record.correlation_id ~= ARGV[3] then
        return redis.error_reply('SUBMISSION_CONFLICT')
    end

    if correlation and correlation ~= ARGV[4] then
        return redis.error_reply('CORRELATION_CONFLICT')
    end

    if not correlation then
        redis.call('SET', KEYS[3], ARGV[4])
    else
        redis.call('PERSIST', KEYS[3])
    end

    return {record.message_id, record.correlation_id, 'existing'}
end

if correlation and correlation ~= ARGV[4] then
    return redis.error_reply('CORRELATION_CONFLICT')
end

-- All ordinary validation is complete before writes.
-- XADD first, so a stream-command failure cannot leave a new
-- submission record.
local message_id = redis.call(
    'XADD', KEYS[2], '*', 'event', ARGV[2]
)

if not correlation then
    redis.call('SET', KEYS[3], ARGV[4])
else
    redis.call('PERSIST', KEYS[3])
end

local record = {
    fingerprint = ARGV[1],
    message_id = message_id,
    correlation_id = ARGV[3]
}

redis.call('SET', KEYS[1], cjson.encode(record))

return {message_id, ARGV[3], 'created'}
"""


@dataclass(frozen=True)
class SubmissionResult:
    message_id: str
    correlation_id: UUID
    created: bool


def fingerprint(data: dict[str, Any]) -> str:
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


async def submit_once(
    redis,
    *,
    stream: str,
    run_id: str,
    fingerprint_value: str,
    event,
    correlation_record,
) -> SubmissionResult:
    _safe_path(Path("."), run_id, "submission")

    if str(event.correlation_id) != str(correlation_record.correlation_id):
        raise ValueError("Event and correlation record identities differ")

    submission_key = f"biotoxbridge:submission:{run_id}"
    correlation_key = f"oaf:correlation:{event.correlation_id}"

    result = await redis.eval(
        SUBMIT_SCRIPT,
        3,
        submission_key,
        stream,
        correlation_key,
        fingerprint_value,
        event.model_dump_json(),
        str(event.correlation_id),
        correlation_record.model_dump_json(),
    )

    return SubmissionResult(
        message_id=_decode(result[0]),
        correlation_id=UUID(_decode(result[1])),
        created=_decode(result[2]) == "created",
    )
