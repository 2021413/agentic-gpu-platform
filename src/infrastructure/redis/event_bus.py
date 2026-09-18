"""Redis Streams event bus (spec sections 15 and 21).

Streams rather than pub/sub: pub/sub drops whatever arrives while a subscriber
is reconnecting, and an SSE client that misses the ``run.completed`` event hangs
until it times out. A stream keeps entries, so a reader is defined by a
position rather than by luck.

Two streams receive every event: one global, for projections and audit tailing,
and one per run, so following a run is a single blocking read rather than a
filter over everything the platform is doing.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import timedelta
from typing import Any, cast

from redis.asyncio import Redis

from domain.events.base import DomainEvent
from domain.value_objects.identifiers import RunId
from infrastructure.redis.codecs import (
    as_text,
    decode_event,
    encode_event,
    is_terminal_for_run,
    run_id_of,
    text_mapping,
)
from infrastructure.redis.keys import Keyspace

__all__ = [
    "DEFAULT_BLOCK",
    "DEFAULT_MAX_STREAM_LENGTH",
    "DEFAULT_RUN_STREAM_TTL",
    "RedisEventBus",
    "RunEventStream",
]

_log = logging.getLogger(__name__)

DEFAULT_MAX_STREAM_LENGTH = 10_000
"""Entries kept per stream. Trimming is approximate: exact trimming costs more
than it buys, and PostgreSQL — not Redis — is the durable record of events."""

DEFAULT_RUN_STREAM_TTL = timedelta(hours=24)
"""A run's stream outlives the run long enough for a late client to catch up,
then disappears on its own. Without it, every run leaks a key forever."""

DEFAULT_BLOCK = timedelta(seconds=1)
"""How long a reader parks inside Redis before looping. Long enough that an idle
subscription is nearly free, short enough that cancellation is felt quickly."""


class RedisEventBus:
    """Publishes domain events to Redis Streams and lets clients follow a run."""

    __slots__ = ("_block_ms", "_keys", "_max_length", "_redis", "_run_stream_ttl_ms")

    def __init__(
        self,
        redis: Redis,
        *,
        keys: Keyspace | None = None,
        max_stream_length: int = DEFAULT_MAX_STREAM_LENGTH,
        run_stream_ttl: timedelta = DEFAULT_RUN_STREAM_TTL,
        block: timedelta = DEFAULT_BLOCK,
    ) -> None:
        self._redis = redis
        self._keys = keys or Keyspace()
        self._max_length = max_stream_length
        self._run_stream_ttl_ms = max(int(run_stream_ttl.total_seconds() * 1000), 1)
        self._block_ms = max(int(block.total_seconds() * 1000), 1)

    async def publish(self, events: Sequence[DomainEvent]) -> None:
        """Publish a batch in one round trip.

        The batch is deliberately not transactional: these events describe
        something that already happened and was already committed, so a partial
        publication is a delivery problem, not a consistency one — and the
        consumers are idempotent.
        """
        if not events:
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for event in events:
                fields = cast("Any", encode_event(event))
                pipe.xadd(
                    self._keys.event_stream,
                    fields,
                    maxlen=self._max_length,
                    approximate=True,
                )
                run_id = run_id_of(event)
                if run_id is None:
                    continue
                run_stream = self._keys.run_stream(run_id)
                pipe.xadd(run_stream, fields, maxlen=self._max_length, approximate=True)
                pipe.pexpire(run_stream, self._run_stream_ttl_ms)
            await pipe.execute()

    def subscribe(self, run_id: RunId) -> AsyncIterator[DomainEvent]:
        """Stream events for one run, starting from now."""
        return RunEventStream(
            self._redis,
            self._keys.run_stream(run_id),
            block_ms=self._block_ms,
            batch=self._max_length,
        )


class RunEventStream:
    """One subscriber's cursor over a run's stream.

    A class rather than an async generator so the starting position can be
    pinned before the first ``__anext__``: ``ready()`` records the stream's last
    entry id, and everything published afterwards is guaranteed to be seen. A
    generator would only resolve its position on first iteration, which is a
    race a caller has no way to close.
    """

    __slots__ = ("_batch", "_block_ms", "_buffer", "_closed", "_key", "_last_id", "_redis")

    def __init__(self, redis: Redis, key: str, *, block_ms: int, batch: int) -> None:
        self._redis = redis
        self._key = key
        self._block_ms = block_ms
        self._batch = batch
        self._last_id: str | None = None
        self._buffer: list[DomainEvent] = []
        self._closed = False

    async def ready(self) -> None:
        """Pin the starting position. Idempotent, and implied by iterating."""
        if self._last_id is not None:
            return
        entries = cast("Any", await self._redis.xrevrange(self._key, count=1))
        self._last_id = as_text(entries[0][0]) if entries else "0-0"

    def __aiter__(self) -> AsyncIterator[DomainEvent]:
        return self

    async def __anext__(self) -> DomainEvent:
        while True:
            if self._buffer:
                return self._buffer.pop(0)
            if self._closed:
                raise StopAsyncIteration
            await self._read()

    async def aclose(self) -> None:
        """Stop the subscription. The caller's disconnect handler calls this."""
        self._closed = True
        self._buffer.clear()

    async def _read(self) -> None:
        await self.ready()
        response = cast(
            "Any",
            await self._redis.xread(
                {self._key: cast("Any", self._last_id)},
                count=self._batch,
                block=self._block_ms,
            ),
        )
        if not response:
            return
        for _stream, entries in response:
            for entry_id, fields in entries:
                self._last_id = as_text(entry_id)
                self._accept(text_mapping(fields))

    def _accept(self, fields: Mapping[str, str]) -> None:
        event = decode_event(fields)
        if event is None:
            # Written by a newer deployment than this reader understands.
            _log.warning("dropping unknown event %r from %s", fields.get("name"), self._key)
            return
        self._buffer.append(event)
        if is_terminal_for_run(event):
            # Deliver it, then stop: there will never be another event for this
            # run, and a subscriber left blocking on a finished run is a leak.
            self._closed = True
