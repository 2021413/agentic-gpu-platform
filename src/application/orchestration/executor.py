"""The job execution loop (spec sections 12, 26 and 36).

The orchestrator process is itself a queue consumer: it claims jobs, holds a
lease while it works, and renews that lease until it is done. Concurrency is
bounded by a semaphore — unlimited task creation is how an async service turns a
traffic spike into an outage.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from application.orchestration.orchestrator import RunOrchestrator
from domain.entities.job import Job
from domain.enums import JobType
from domain.exceptions import JobLeaseExpiredError
from domain.ports.clock import Clock
from domain.ports.job_queue import JobQueue
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.lease import Lease

__all__ = ["ExecutorConfig", "JobExecutor"]

_log = logging.getLogger(__name__)

ALL_JOB_TYPES: tuple[JobType, ...] = tuple(JobType)


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    concurrency: int = 8
    lease_duration: timedelta = timedelta(seconds=120)
    poll_interval: timedelta = timedelta(seconds=1)
    idle_backoff: timedelta = timedelta(seconds=2)

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")

    @property
    def renew_interval(self) -> timedelta:
        """Renew well before expiry, so one slow tick cannot lose the job."""
        return self.lease_duration / 3


class JobExecutor:
    """Claims and executes jobs until stopped."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        orchestrator: RunOrchestrator,
        clock: Clock,
        executor_id: WorkerId,
        config: ExecutorConfig | None = None,
        job_types: Sequence[JobType] = ALL_JOB_TYPES,
    ) -> None:
        self._queue = queue
        self._orchestrator = orchestrator
        self._clock = clock
        self._executor_id = executor_id
        self._config = config or ExecutorConfig()
        self._job_types = tuple(job_types)
        self._semaphore = asyncio.Semaphore(self._config.concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._stopping = asyncio.Event()

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    async def claim_one(self) -> tuple[Job, Lease] | None:
        return await self._queue.claim(
            consumer=self._executor_id,
            job_types=self._job_types,
            lease_duration=self._config.lease_duration,
            now=self._clock.now(),
        )

    async def run_once(self) -> bool:
        """Claim and execute at most one job. Returns False when nothing was due.

        Useful on its own in tests: the whole workflow can be driven step by
        step without ever starting a background loop.
        """
        await self._semaphore.acquire()
        try:
            claimed = await self.claim_one()
        except Exception:
            self._semaphore.release()
            raise
        if claimed is None:
            self._semaphore.release()
            return False

        job, lease = claimed
        try:
            await self._execute_with_renewal(job, lease)
        finally:
            self._semaphore.release()
        return True

    async def run_forever(self) -> None:
        """Poll until stopped, then let in-flight jobs finish."""
        self._stopping.clear()
        try:
            while not self._stopping.is_set():
                await self._semaphore.acquire()
                claimed: tuple[Job, Lease] | None = None
                try:
                    claimed = await self.claim_one()
                except Exception:
                    _log.exception("failed to claim a job; backing off")
                if claimed is None:
                    self._semaphore.release()
                    await self._sleep(self._config.idle_backoff)
                    continue
                job, lease = claimed
                self._spawn(job, lease)
        finally:
            await self.drain()

    async def stop(self) -> None:
        self._stopping.set()

    async def drain(self) -> None:
        """Let accepted jobs finish, as a draining worker would."""
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------
    def _spawn(self, job: Job, lease: Lease) -> None:
        async def _runner() -> None:
            try:
                await self._execute_with_renewal(job, lease)
            finally:
                self._semaphore.release()

        task = asyncio.create_task(_runner(), name=f"job-{job.id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute_with_renewal(self, job: Job, lease: Lease) -> None:
        """Run the job while a companion task keeps its lease alive."""
        renewer = asyncio.create_task(self._renew(job, lease), name=f"lease-{job.id}")
        try:
            await self._orchestrator.execute(job, lease)
        finally:
            renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewer

    async def _renew(self, job: Job, lease: Lease) -> None:
        interval = self._config.renew_interval.total_seconds()
        while True:
            await asyncio.sleep(interval)
            try:
                await self._queue.renew(
                    job_id=job.id,
                    token=lease.token,
                    duration=self._config.lease_duration,
                    now=self._clock.now(),
                )
            except JobLeaseExpiredError:
                # The job was reassigned while we were working. Stop renewing;
                # the result we eventually produce will be rejected by the token
                # check, which is exactly the protection we want.
                _log.warning("lost the lease on job %s; it has been reassigned", job.id)
                return
            except Exception:
                _log.exception("failed to renew the lease on job %s", job.id)

    async def _sleep(self, delay: timedelta) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=delay.total_seconds())
