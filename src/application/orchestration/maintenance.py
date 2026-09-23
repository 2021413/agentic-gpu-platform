"""Background maintenance loops.

These are what make failure recovery automatic rather than aspirational: a
worker that stops answering is declared unavailable, and the jobs it was holding
become claimable again once their lease lapses.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from application.orchestration.orchestrator import RunOrchestrator
from application.ports import MetricsRecorder, UnitOfWorkFactory
from application.services.event_publisher import commit_and_publish
from application.use_cases.workers import ReapStaleWorkersUseCase
from domain.enums import JobType
from domain.ports.clock import Clock
from domain.ports.event_bus import EventBus
from domain.ports.job_queue import JobQueue
from domain.ports.worker_registry import WorkerRegistry
from domain.value_objects.identifiers import JobId

__all__ = ["MaintenanceConfig", "MaintenanceLoop"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaintenanceConfig:
    interval: timedelta = timedelta(seconds=10)
    reclaim_batch: int = 100


class MaintenanceLoop:
    """Reaps silent workers and requeues jobs whose lease expired."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        uow_factory: UnitOfWorkFactory,
        bus: EventBus,
        clock: Clock,
        reaper: ReapStaleWorkersUseCase,
        orchestrator: RunOrchestrator | None = None,
        registry: WorkerRegistry | None = None,
        metrics: MetricsRecorder | None = None,
        config: MaintenanceConfig | None = None,
    ) -> None:
        self._queue = queue
        self._uow_factory = uow_factory
        self._bus = bus
        self._clock = clock
        self._reaper = reaper
        self._orchestrator = orchestrator
        self._registry = registry
        self._metrics = metrics
        self._config = config or MaintenanceConfig()
        self._stopping = asyncio.Event()

    async def tick(self) -> Sequence[JobId]:
        """One maintenance pass. Returns the jobs that were made retryable."""
        if self._orchestrator is not None:
            # Creating a run and scheduling it are two different things: the API
            # persists the run and answers immediately rather than holding a
            # client on a GPU. Without this sweep a run created over HTTP stayed
            # in CREATED forever, and the API could accept work it never did.
            started = await self._orchestrator.advance_stalled_runs()
            if started:
                _log.info("advanced %d run(s) that had stopped without finishing", len(started))

        reaped = await self._reaper.execute()
        if reaped:
            _log.info("declared %d worker(s) unavailable after heartbeat timeout", len(reaped))

        expired = await self._queue.reclaim_expired(
            now=self._clock.now(), limit=self._config.reclaim_batch
        )
        requeued: list[JobId] = []
        for job_id in expired:
            if await self._requeue(job_id):
                requeued.append(job_id)

        await self._observe()
        return requeued

    async def _observe(self) -> None:
        """Publish the five gauges the platform declares.

        They were declared on day one and nothing ever set them. That is not a
        gap that shows up as a missing metric: an unfed Prometheus gauge reads
        **zero**, so a dashboard wired to these reported "no active runs, no
        registered workers" with exactly the confidence of a real measurement.

        This loop is the right place because it already runs on a timer and
        already holds every number: a gauge is a sample of the present, not an
        event, so it belongs with the other periodic work rather than scattered
        across the code paths that happen to change the underlying quantity.

        Telemetry is never load-bearing here. A registry that cannot be read is
        a reason to skip the sample, not to fail a maintenance pass that has
        just requeued real work.
        """
        if self._metrics is None:
            return
        try:
            async with self._uow_factory() as uow:
                active_runs = len(await uow.runs.list_active())
            self._metrics.gauge("active_runs", active_runs)

            for job_type in JobType:
                self._metrics.gauge(
                    "queued_jobs", await self._queue.depth(job_type), job_type=job_type.value
                )

            if self._registry is not None:
                workers = await self._registry.list_all()
                available = await self._registry.list_available()
                self._metrics.gauge("registered_workers", len(workers))
                self._metrics.gauge("healthy_workers", len(available))
                self._metrics.gauge("active_jobs", sum(w.active_jobs for w in workers))
        except Exception:
            _log.warning("could not sample the platform gauges", exc_info=True)

    async def run_forever(self) -> None:
        self._stopping.clear()
        while not self._stopping.is_set():
            try:
                await self.tick()
            except Exception:
                _log.exception("maintenance tick failed; continuing")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self._config.interval.total_seconds()
                )

    async def stop(self) -> None:
        self._stopping.set()

    async def _requeue(self, job_id: JobId) -> bool:
        """Give an abandoned job another attempt, or let it die honestly."""
        now = self._clock.now()
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None or job.status.is_terminal:
                return False
            retryable = job.expire_lease(now)
            if retryable:
                job.requeue(now=now, reason="lease expired: the worker stopped reporting")
            await uow.jobs.update(job)
            uow.collect(job)
            await commit_and_publish(uow, self._bus)

        if retryable:
            await self._queue.enqueue(job)
            return True

        # Out of attempts: the run must learn about it rather than wait forever.
        if self._orchestrator is not None:
            await self._orchestrator.handle_dead_job(job)
        return False
