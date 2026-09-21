"""Acquiring a GPU worker for one job.

The pool is asked on every single job. Nothing is cached, nothing is pinned to
a project: that is precisely what lets workers appear and disappear mid-run
without restarting anything.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from domain.entities.worker import Worker
from domain.exceptions import NoCompatibleWorkerError
from domain.ports.llm_provider import LLMProvider, LLMProviderFactory
from domain.ports.worker_registry import WorkerRegistry
from domain.ports.worker_scheduler import WorkerScheduler
from domain.value_objects.worker import JobRequirements

__all__ = ["AcquiredWorker", "WorkerPool"]


@dataclass(frozen=True, slots=True)
class AcquiredWorker:
    """A worker with a reserved slot and a provider bound to its endpoint."""

    worker: Worker
    provider: LLMProvider

    @property
    def model_id(self) -> str:
        return self.worker.capabilities.model_id


class WorkerPool:
    """Turns "I need a worker for this job" into a reserved, usable worker."""

    __slots__ = ("_factory", "_registry", "_scheduler")

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        scheduler: WorkerScheduler,
        factory: LLMProviderFactory,
    ) -> None:
        self._registry = registry
        self._scheduler = scheduler
        self._factory = factory

    async def try_acquire(self, requirements: JobRequirements) -> AcquiredWorker | None:
        """Reserve a compatible worker, or ``None`` if none is free right now.

        Returning ``None`` is an ordinary outcome in an elastic pool: the caller
        re-queues the job with a backoff instead of failing the run, because a
        worker may register a second later.
        """
        available = await self._registry.list_available()
        chosen = self._scheduler.select(candidates=available, requirements=requirements)
        if chosen is None:
            return None

        # The reservation is what keeps scheduling correct between heartbeats,
        # which are far too coarse to reflect second-by-second occupancy.
        chosen.reserve_slot(requirements)
        await self._registry.update(chosen)
        provider = self._factory.for_endpoint(
            chosen.endpoint, model_id=chosen.capabilities.model_id
        )
        return AcquiredWorker(worker=chosen, provider=provider)

    async def release(self, acquired: AcquiredWorker) -> None:
        acquired.worker.release_slot()
        await self._registry.update(acquired.worker)

    @asynccontextmanager
    async def acquire(self, requirements: JobRequirements) -> AsyncIterator[AcquiredWorker]:
        """Reserve for the duration of the block, releasing whatever happens.

        Raises ``NoCompatibleWorkerError`` when the pool cannot serve the job;
        the caller decides whether that means "wait" or "give up".
        """
        acquired = await self.try_acquire(requirements)
        if acquired is None:
            raise NoCompatibleWorkerError(
                "no compatible worker is available",
                role=str(requirements.role),
                model_id=requirements.model_id,
            )
        try:
            yield acquired
        finally:
            await self.release(acquired)

    async def capacity_for(self, requirements: JobRequirements) -> int:
        """Free slots across the pool, used to size candidate fan-out."""
        available = await self._registry.list_available()
        return sum(w.available_slots for w in available if w.can_accept(requirements))

    async def prompt_budget(self, requirements: JobRequirements) -> int | None:
        """How large a prompt the fleet can actually take for this job.

        The roomiest worker sets it, because a prompt only has to fit
        *somewhere* and the scheduler will route it there; the narrowest must
        not cap what the others could hold.

        ``None`` when nothing registered can serve the role. That has to stay
        distinguishable from a number: the caller then falls back to its own
        configured ceiling rather than to an invented one, which is exactly how
        a 262144 declaration ended up in front of a 16384 engine.

        Load is ignored on purpose. This answers "how much room exists", not
        "who is free now" — a busy worker still bounds what is worth building.
        """
        workers = [
            w
            for w in await self._registry.list_available()
            if w.capabilities.supports_role(requirements.role)
        ]
        if not workers:
            return None
        return max(
            w.capabilities.usable_prompt_tokens(
                reserved_output_tokens=requirements.reserved_output_tokens
            )
            for w in workers
        )
