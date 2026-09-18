"""Worker management use cases (spec sections 3.3, 19 and 25).

These are the operations that make the pool elastic: a worker announces itself,
proves it is alive, stops taking work, and leaves — all at runtime, with no
restart of the orchestrator and no change to project code.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from application.dto.commands import (
    DeregisterWorkerCommand,
    DrainWorkerCommand,
    HeartbeatCommand,
    RegisterWorkerCommand,
)
from application.dto.views import WorkerView
from domain.entities.worker import Worker
from domain.enums import WorkerStatus
from domain.exceptions import EntityNotFoundError
from domain.ports.clock import Clock, IdGenerator
from domain.ports.event_bus import EventBus
from domain.ports.worker_registry import WorkerRegistry
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import WorkerCapabilities, WorkerEndpoint

__all__ = [
    "DeregisterWorkerUseCase",
    "DrainWorkerUseCase",
    "HeartbeatUseCase",
    "ListWorkersUseCase",
    "ReapStaleWorkersUseCase",
    "RegisterWorkerUseCase",
]


class RegisterWorkerUseCase:
    """Admit a worker into the pool.

    Registration is idempotent on the worker id so a worker that restarts, or
    that retries a timed-out call, refreshes its entry instead of creating a
    twin that would double the apparent capacity.
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        bus: EventBus,
        clock: Clock,
        ids: IdGenerator,
        heartbeat_ttl: timedelta,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._clock = clock
        self._ids = ids
        self._ttl = heartbeat_ttl

    async def execute(self, command: RegisterWorkerCommand) -> WorkerView:
        now = self._clock.now()
        capabilities = WorkerCapabilities(
            model_id=command.model_id,
            context_length=command.context_length,
            max_concurrency=command.max_concurrency,
            supported_roles=command.supported_roles,
            gpu=command.gpu,
            supports_tools=command.supports_tools,
            supports_json_schema=command.supports_json_schema,
            metadata=dict(command.metadata),
        )
        endpoint = WorkerEndpoint(command.endpoint)
        worker_id = command.worker_id or self._ids.next_id(WorkerId)

        existing = await self._registry.get(worker_id) if command.worker_id else None
        if existing is not None:
            existing.heartbeat(now=now, capabilities=capabilities, status=WorkerStatus.READY)
            await self._registry.register(existing, ttl=self._ttl)
            await self._bus.publish(existing.pull_events())
            return WorkerView.of(existing)

        worker = Worker.register(
            worker_id=worker_id,
            endpoint=endpoint,
            capabilities=capabilities,
            now=now,
            metadata=command.metadata,
        )
        await self._registry.register(worker, ttl=self._ttl)
        await self._bus.publish(worker.pull_events())
        return WorkerView.of(worker)


class HeartbeatUseCase:
    """Refresh liveness and self-reported load.

    A heartbeat from a worker previously declared unavailable readmits it: a
    network partition must not permanently remove a healthy GPU.
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        clock: Clock,
        heartbeat_ttl: timedelta,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._ttl = heartbeat_ttl

    async def execute(self, command: HeartbeatCommand) -> WorkerView:
        worker = await self._registry.get(command.worker_id)
        if worker is None:
            raise EntityNotFoundError("Worker", command.worker_id)

        now = self._clock.now()
        worker.heartbeat(
            now=now,
            load=command.load,
            status=WorkerStatus.DRAINING if command.draining else None,
        )
        await self._registry.heartbeat(command.worker_id, load=command.load, at=now, ttl=self._ttl)
        await self._registry.update(worker)
        return WorkerView.of(worker)


class DrainWorkerUseCase:
    """Stop sending work to a worker while letting it finish what it holds."""

    def __init__(self, *, registry: WorkerRegistry, bus: EventBus, clock: Clock) -> None:
        self._registry = registry
        self._bus = bus
        self._clock = clock

    async def execute(self, command: DrainWorkerCommand) -> WorkerView:
        worker = await self._registry.get(command.worker_id)
        if worker is None:
            raise EntityNotFoundError("Worker", command.worker_id)
        worker.start_draining(self._clock.now())
        await self._registry.update(worker)
        await self._bus.publish(worker.pull_events())
        return WorkerView.of(worker)


class DeregisterWorkerUseCase:
    """Remove a worker. Jobs it still held are reclaimed through lease expiry."""

    def __init__(self, *, registry: WorkerRegistry, bus: EventBus, clock: Clock) -> None:
        self._registry = registry
        self._bus = bus
        self._clock = clock

    async def execute(self, command: DeregisterWorkerCommand) -> None:
        worker = await self._registry.get(command.worker_id)
        if worker is None:
            # Deregistration is idempotent: a worker that already left, or that
            # retries the call, must not turn into a 404 storm.
            return
        worker.deregister(now=self._clock.now(), graceful=command.graceful)
        await self._registry.deregister(command.worker_id)
        await self._bus.publish(worker.pull_events())


class ListWorkersUseCase:
    def __init__(self, *, registry: WorkerRegistry) -> None:
        self._registry = registry

    async def execute(self, *, only_available: bool = False) -> Sequence[WorkerView]:
        workers = (
            await self._registry.list_available()
            if only_available
            else await self._registry.list_all()
        )
        return [WorkerView.of(w) for w in workers]


class ReapStaleWorkersUseCase:
    """Declare silent workers unavailable so their jobs can be retried elsewhere.

    The events the registry produced are published here. A fleet losing workers
    is precisely what an operator needs to see, so swallowing those events would
    make the most interesting failure the quietest one.
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        clock: Clock,
        heartbeat_timeout: timedelta,
        bus: EventBus | None = None,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._timeout = heartbeat_timeout
        self._bus = bus

    async def execute(self) -> Sequence[WorkerId]:
        reaped = await self._registry.reap_stale(
            now=self._clock.now(), heartbeat_timeout=self._timeout
        )
        if self._bus is not None:
            for worker in reaped:
                await self._bus.publish(worker.pull_events())
        return [worker.id for worker in reaped]
