"""Redis-backed job queue with leases (spec sections 12, 35 and 36).

The shape of the data is what makes the operations possible:

* one hash per job holds the whole record, so a claim can hand back a real
  ``Job`` without a database round trip;
* one sorted set per job type holds what is claimable, scored so that the
  lowest score is the job to run next — priority first, age as the tie-break;
* one sorted set holds every lease, scored by expiry, which turns "find the
  work a dead worker was holding" into a range query;
* one set per run holds its job ids, which is what makes cancelling a run cheap
  and precise.

Everything that reads before it writes is a Lua script (see
``infrastructure.redis.scripts``), because Redis runs those to completion:
that, and only that, is why two consumers polling at the same instant cannot
walk away with the same job.

Delivery is at-least-once by assumption. Every operation is therefore written
so that replaying it changes nothing: enqueue loses to a live lease and to the
acknowledgement tombstone, acknowledging an unknown job succeeds quietly, and
reporting with a token the job no longer carries is ignored rather than
obeyed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, cast

from redis.asyncio import Redis

from domain.entities.job import Job
from domain.enums import JobStatus, JobType
from domain.exceptions import JobLeaseExpiredError
from domain.value_objects.identifiers import JobId, RunId, WorkerId
from domain.value_objects.lease import Lease, LeaseToken
from infrastructure.redis import scripts
from infrastructure.redis.codecs import (
    as_text,
    job_from_mapping,
    job_to_mapping,
    queue_score,
)
from infrastructure.redis.keys import Keyspace

__all__ = ["DEFAULT_TOMBSTONE_TTL", "RedisJobQueue"]

DEFAULT_TOMBSTONE_TTL = timedelta(hours=1)
"""How long a finished job's record is kept after acknowledgement.

It exists purely to absorb redelivery: a queue message replayed within the hour
finds the tombstone and is refused. Longer would keep dead weight in Redis,
shorter would narrow the window in which at-least-once delivery is actually
survivable.
"""


def _epoch_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _flatten(mapping: dict[str, str]) -> list[str]:
    return [item for pair in mapping.items() for item in pair]


def _pairs(flat: Sequence[object]) -> dict[str, str]:
    """Rebuild a hash from the flat array a Lua ``HGETALL`` returns."""
    text = [as_text(item) for item in flat]
    return dict(zip(text[0::2], text[1::2], strict=True))


class RedisJobQueue:
    """Transport for schedulable work. Durable state lives in PostgreSQL."""

    __slots__ = (
        "_acknowledge",
        "_cancel",
        "_claim",
        "_enqueue",
        "_keys",
        "_reclaim",
        "_redis",
        "_release",
        "_renew",
        "_tombstone_ms",
    )

    def __init__(
        self,
        redis: Redis,
        *,
        keys: Keyspace | None = None,
        tombstone_ttl: timedelta = DEFAULT_TOMBSTONE_TTL,
    ) -> None:
        self._redis = redis
        self._keys = keys or Keyspace()
        self._tombstone_ms = max(int(tombstone_ttl.total_seconds() * 1000), 1)
        self._enqueue = redis.register_script(scripts.ENQUEUE)
        self._claim = redis.register_script(scripts.CLAIM)
        self._renew = redis.register_script(scripts.RENEW)
        self._acknowledge = redis.register_script(scripts.ACKNOWLEDGE)
        self._release = redis.register_script(scripts.RELEASE)
        self._reclaim = redis.register_script(scripts.RECLAIM_EXPIRED)
        self._cancel = redis.register_script(scripts.CANCEL_RUN_JOBS)

    # -- publication ----------------------------------------------------
    async def enqueue(self, job: Job) -> None:
        """Publish a job.

        The stored status is forced to ``QUEUED``: sitting in the ready set is
        what being queued *means* here, and accepting a job that claims
        otherwise would leave the two disagreeing. The caller's entity is
        untouched — the durable record in PostgreSQL remains the authority.
        """
        record = job_to_mapping(job)
        record["status"] = JobStatus.QUEUED.value
        await self._enqueue(
            keys=[
                self._keys.job(job.id),
                self._keys.ready(job.type),
                self._keys.run_jobs(job.run_id),
            ],
            args=[str(job.id), queue_score(job.priority, job.created_at), *_flatten(record)],
        )

    # -- consumption ----------------------------------------------------
    async def claim(
        self,
        *,
        consumer: WorkerId,
        job_types: Sequence[JobType],
        lease_duration: timedelta,
        now: datetime,
    ) -> tuple[Job, Lease] | None:
        """Atomically take the highest-priority eligible job, or ``None``.

        The script returns the job *as it was before the lease*, and the
        transition is then replayed through ``Job.lease_to``. The entity, not
        the Lua, decides what a leased job looks like — the script only
        guarantees that no one else can be doing the same thing at the same
        time. The returned job therefore carries a ``JobLeased`` event in its
        buffer, for the caller to drain with everything else it publishes.
        """
        wanted = list(dict.fromkeys(job_types))
        if not wanted:
            return None
        token = LeaseToken.generate()
        expires_at = now + lease_duration
        raw = cast(
            "Any",
            await self._claim(
                keys=[self._keys.leases, *(self._keys.ready(job_type) for job_type in wanted)],
                args=[
                    self._keys.job_prefix,
                    _epoch_ms(now),
                    _epoch_ms(expires_at),
                    token.value,
                    str(consumer),
                    now.isoformat(),
                    expires_at.isoformat(),
                    self._keys.ready_prefix,
                    self._keys.delayed_prefix,
                    ",".join(job_type.value for job_type in wanted),
                ],
            ),
        )
        if not raw:
            return None
        job = job_from_mapping(_pairs(cast("Sequence[object]", raw)))
        lease = job.lease_to(worker_id=consumer, now=now, duration=lease_duration, token=token)
        return job, lease

    async def renew(
        self, *, job_id: JobId, token: LeaseToken, duration: timedelta, now: datetime
    ) -> Lease:
        """Extend a held lease. Raises ``JobLeaseExpiredError`` if it was lost.

        Losing a lease is not an anomaly to paper over: the job may already be
        running somewhere else, so the caller must stop and drop its result.
        """
        expires_at = now + duration
        result = cast(
            "Any",
            await self._renew(
                keys=[self._keys.job(job_id), self._keys.leases],
                args=[
                    str(job_id),
                    token.value,
                    _epoch_ms(now),
                    _epoch_ms(expires_at),
                    expires_at.isoformat(),
                ],
            ),
        )
        if not result:
            raise JobLeaseExpiredError(job_id)
        acquired_at, holder = result
        return Lease(
            job_id=job_id,
            holder=WorkerId.parse(as_text(holder)),
            token=token,
            acquired_at=datetime.fromisoformat(as_text(acquired_at)),
            expires_at=expires_at,
        )

    async def acknowledge(self, *, job_id: JobId, token: LeaseToken) -> None:
        """Remove a finished job from the queue."""
        await self._acknowledge(
            keys=[self._keys.job(job_id), self._keys.leases],
            args=[
                str(job_id),
                token.value,
                self._keys.run_jobs_prefix,
                self._keys.run_jobs_suffix,
                self._tombstone_ms,
            ],
        )

    async def release(
        self,
        *,
        job_id: JobId,
        token: LeaseToken,
        requeue: bool,
        not_before: datetime | None = None,
    ) -> None:
        """Give a job back, optionally making it claimable again, possibly later.

        The attempt counter is left where the claim put it: the attempt really
        was spent, and hiding that would let a job loop forever between a
        worker that cannot do it and a queue that keeps offering it.

        A deferred requeue lands in the delayed set instead of the ready set,
        and `claim` promotes it when its time comes. It is deliberately not a
        sweeper: promotion happening inside the same script as the claim means
        there is no instant at which a due job belongs to neither set.
        """
        deferred = requeue and not_before is not None
        await self._release(
            keys=[self._keys.job(job_id), self._keys.leases],
            args=[
                str(job_id),
                token.value,
                "2" if deferred else ("1" if requeue else "0"),
                self._keys.ready_prefix,
                _epoch_ms(not_before) if not_before is not None else "0",
                self._keys.delayed_prefix,
            ],
        )

    # -- recovery -------------------------------------------------------
    async def reclaim_expired(self, *, now: datetime, limit: int = 100) -> Sequence[JobId]:
        """Return jobs whose lease lapsed so the orchestrator can retry them.

        Jobs with attempts left go straight back into the ready set at their
        original position; the rest are marked dead in place. Either way the
        ids are returned, because the durable record in PostgreSQL still has to
        be told what happened to them.
        """
        reclaimed = cast(
            "Any",
            await self._reclaim(
                keys=[self._keys.leases],
                args=[
                    _epoch_ms(now),
                    limit,
                    self._keys.job_prefix,
                    self._keys.ready_prefix,
                    now.isoformat(),
                    self._tombstone_ms,
                ],
            ),
        )
        return [JobId.parse(as_text(job_id)) for job_id in reclaimed]

    async def cancel_run_jobs(self, run_id: RunId) -> Sequence[JobId]:
        """Drop every pending job of a run; in-flight ones are cancelled by lease.

        Returns every job it actually stopped, in-flight ones included: they no
        longer hold a valid lease, so their worker will find out on its next
        renewal, and the caller needs the full list to reconcile the run.
        """
        cancelled = cast(
            "Any",
            await self._cancel(
                keys=[self._keys.run_jobs(run_id), self._keys.leases],
                args=[
                    self._keys.job_prefix,
                    self._keys.ready_prefix,
                    self._tombstone_ms,
                ],
            ),
        )
        return [JobId.parse(as_text(job_id)) for job_id in cancelled]

    # -- metrics --------------------------------------------------------
    async def depth(self, job_type: JobType | None = None) -> int:
        """Claimable jobs, which is what backlog means for autoscaling.

        In-flight jobs are excluded on purpose: they already have a worker, and
        counting them would make the queue look deep precisely when it is being
        drained fastest.
        """
        # Jobs waiting out a retry delay are counted too. They are queued, they
        # have no worker, and they are exactly what somebody reading this number
        # needs to see — a backlog that is invisible while it waits is worse
        # than no number at all.
        if job_type is not None:
            async with self._redis.pipeline(transaction=False) as pipe:
                pipe.zcard(self._keys.ready(job_type))
                pipe.zcard(self._keys.delayed(job_type))
                counts = cast("list[int]", await pipe.execute())
            return sum(counts)
        async with self._redis.pipeline(transaction=False) as pipe:
            for known in JobType:
                pipe.zcard(self._keys.ready(known))
                pipe.zcard(self._keys.delayed(known))
            depths = cast("list[int]", await pipe.execute())
        return sum(depths)
