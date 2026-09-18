"""Job queue adapters, Redis-backed and in-memory.

Both implement ``domain.ports.JobQueue`` with the same semantics, which is what
lets the platform run locally without Docker and still be tested against the
behaviour it will have in production. ``_ports_are_satisfied`` below is not
dead code: structural typing means a signature that drifts away from a port
would otherwise go unnoticed until the first caller tripped over it at runtime,
so the type checker is asked the question here instead.
"""

from __future__ import annotations

from domain.ports.event_bus import EventBus
from domain.ports.job_queue import JobQueue
from domain.ports.worker_registry import WorkerRegistry
from infrastructure.queue.in_memory import (
    InMemoryEventBus,
    InMemoryJobQueue,
    InMemoryRunEventStream,
    InMemoryWorkerRegistry,
)
from infrastructure.queue.redis_job_queue import RedisJobQueue
from infrastructure.redis.event_bus import RedisEventBus
from infrastructure.redis.worker_registry import RedisWorkerRegistry

__all__ = [
    "InMemoryEventBus",
    "InMemoryJobQueue",
    "InMemoryRunEventStream",
    "InMemoryWorkerRegistry",
    "RedisJobQueue",
]


def _ports_are_satisfied(
    *,
    redis_queue: RedisJobQueue,
    memory_queue: InMemoryJobQueue,
    redis_registry: RedisWorkerRegistry,
    memory_registry: InMemoryWorkerRegistry,
    redis_bus: RedisEventBus,
    memory_bus: InMemoryEventBus,
) -> tuple[JobQueue, JobQueue, WorkerRegistry, WorkerRegistry, EventBus, EventBus]:
    """Type-level proof that all six adapters still implement their port."""
    return (
        redis_queue,
        memory_queue,
        redis_registry,
        memory_registry,
        redis_bus,
        memory_bus,
    )
