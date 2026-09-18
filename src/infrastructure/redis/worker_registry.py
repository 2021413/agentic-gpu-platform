"""Redis-backed worker registry (spec sections 4, 14 and 25).

The pool is discovered, never configured, so this is the only thing that knows
which GPUs exist. Two decisions shape the whole adapter:

* **the record outlives the heartbeat TTL.** A key that vanished the moment a
  worker went quiet would make an outage invisible: nothing left to mark
  ``OFFLINE``, nothing to report to the orchestrator, and the jobs that worker
  was holding would only surface later through lease expiry. The hash is kept
  for ``retention_factor`` times the TTL so the disappearance is observable,
  then Redis collects it on its own;
* **staleness is decided from the stored timestamp, not from key expiry.**
  ``reap_stale`` is handed a ``now``, exactly like the domain entity it
  delegates to, which keeps reaping deterministic and testable against a fake
  clock instead of a real sleep.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, cast

from redis.asyncio import Redis

from domain.entities.worker import Worker
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import WorkerLoad
from infrastructure.redis import scripts
from infrastructure.redis.codecs import (
    as_text,
    text_mapping,
    worker_from_mapping,
    worker_to_mapping,
)
from infrastructure.redis.keys import Keyspace

__all__ = ["DEFAULT_RETENTION_FACTOR", "RedisWorkerRegistry"]

DEFAULT_RETENTION_FACTOR = 10
"""How many heartbeat TTLs a silent worker's record survives before Redis drops it."""

_KEEP_TTL = -1
"""Sentinel for the update script: rewrite fields, leave the expiry as it is."""


def _milliseconds(delta: timedelta) -> int:
    """Whole milliseconds, never zero: a TTL of 0 would mean 'expire now'."""
    return max(int(delta.total_seconds() * 1000), 1)


def _flatten(mapping: dict[str, str]) -> list[str]:
    return [item for pair in mapping.items() for item in pair]


class RedisWorkerRegistry:
    """Registration, heartbeats and liveness for the GPU pool, backed by Redis."""

    __slots__ = ("_keys", "_redis", "_retention_factor", "_update")

    def __init__(
        self,
        redis: Redis,
        *,
        keys: Keyspace | None = None,
        retention_factor: int = DEFAULT_RETENTION_FACTOR,
    ) -> None:
        if retention_factor < 1:
            raise ValueError("retention_factor must be at least 1")
        self._redis = redis
        self._keys = keys or Keyspace()
        self._retention_factor = retention_factor
        self._update = redis.register_script(scripts.WORKER_UPDATE)

    # -- registration ---------------------------------------------------
    async def register(self, worker: Worker, *, ttl: timedelta) -> None:
        """Add or refresh a registration.

        Registering twice overwrites rather than conflicts, which is what makes
        a worker restart — or a retried registration call — safe to replay.
        """
        key = self._keys.worker(worker.id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=cast("Any", worker_to_mapping(worker)))
            pipe.pexpire(key, self._retention_ms(ttl))
            pipe.sadd(self._keys.worker_index, str(worker.id))
            await pipe.execute()

    async def heartbeat(
        self, worker_id: WorkerId, *, load: WorkerLoad, at: datetime, ttl: timedelta
    ) -> bool:
        """Refresh liveness and self-reported load.

        Read-modify-write on purpose: the transition rules (a worker declared
        unavailable is readmitted, BUSY follows capacity) belong to the entity,
        not to a Lua script. The race that usually condemns read-modify-write
        does not exist here — a worker is the only writer of its own record —
        and the final write refuses to recreate a key that disappeared in
        between.
        """
        raw = await self._redis.hgetall(self._keys.worker(worker_id))
        if not raw:
            await self._redis.srem(self._keys.worker_index, str(worker_id))
            return False
        worker = worker_from_mapping(text_mapping(raw))
        worker.heartbeat(now=at, load=load)
        return await self._write(
            worker_id,
            {
                "status": worker.status.value,
                "last_heartbeat_at": worker.last_heartbeat_at.isoformat(),
                "active_jobs": str(worker.load.active_jobs),
                "queued_jobs": str(worker.load.queued_jobs),
            },
            ttl_ms=self._retention_ms(ttl),
        )

    async def update(self, worker: Worker) -> None:
        """Persist a state change made on the entity (drain, unhealthy, load)."""
        await self._write(worker.id, worker_to_mapping(worker), ttl_ms=_KEEP_TTL)

    async def deregister(self, worker_id: WorkerId) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(self._keys.worker(worker_id))
            pipe.srem(self._keys.worker_index, str(worker_id))
            await pipe.execute()

    # -- queries --------------------------------------------------------
    async def get(self, worker_id: WorkerId) -> Worker | None:
        raw = await self._redis.hgetall(self._keys.worker(worker_id))
        return worker_from_mapping(text_mapping(raw)) if raw else None

    async def list_all(self) -> Sequence[Worker]:
        """Every known worker, including draining and unavailable ones.

        Also the point where the index self-heals: an id whose record aged out
        is dropped, so a long-lived deployment does not accumulate ghosts.
        """
        ids = sorted(
            as_text(member) for member in await self._redis.smembers(self._keys.worker_index)
        )
        if not ids:
            return ()
        async with self._redis.pipeline(transaction=False) as pipe:
            for worker_id in ids:
                pipe.hgetall(self._keys.worker(worker_id))
            records = cast("list[Mapping[Any, Any]]", await pipe.execute())
        workers = [worker_from_mapping(text_mapping(raw)) for raw in records if raw]
        missing = [worker_id for worker_id, raw in zip(ids, records, strict=True) if not raw]
        if missing:
            await self._redis.srem(self._keys.worker_index, *missing)
        return workers

    async def list_available(self) -> Sequence[Worker]:
        """Workers that may receive new jobs right now.

        Availability is read from the stored status and load rather than from
        key expiry, so this answers the same way as the in-memory adapter and
        cannot depend on how long a test took to run.
        """
        return [
            worker
            for worker in await self.list_all()
            if worker.status.accepts_new_jobs and worker.available_slots > 0
        ]

    # -- liveness -------------------------------------------------------
    async def reap_stale(self, *, now: datetime, heartbeat_timeout: timedelta) -> Sequence[Worker]:
        """Mark silent workers unavailable and return them, events included.

        The entities come back with ``WorkerStatusChanged`` and
        ``WorkerUnavailable`` still in their buffer: only the caller knows when
        its transaction commits, so only the caller can decide when to publish
        them. Draining the buffer here would silence a fleet failure.

        Safe to run from several orchestrators at once: marking an already
        offline worker offline changes nothing, and a worker is only returned
        if this call is the one that transitioned it — and if its record was
        still there to be written.
        """
        reaped: list[Worker] = []
        for worker in await self.list_all():
            if not worker.status.is_live or not worker.is_stale(now, heartbeat_timeout):
                continue
            worker.mark_unavailable(now=now, reason="heartbeat timeout")
            if await self._write(worker.id, {"status": worker.status.value}, ttl_ms=_KEEP_TTL):
                reaped.append(worker)
        return reaped

    # -- internals ------------------------------------------------------
    def _retention_ms(self, ttl: timedelta) -> int:
        return _milliseconds(ttl) * self._retention_factor

    async def _write(self, worker_id: WorkerId, fields: dict[str, str], *, ttl_ms: int) -> bool:
        """Rewrite fields of an existing record; ``False`` when it is already gone."""
        args: Iterable[Any] = [ttl_ms, *_flatten(fields)]
        written = await self._update(keys=[self._keys.worker(worker_id)], args=list(args))
        return bool(written)
