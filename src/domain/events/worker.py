"""Events describing GPU worker lifecycle (spec section 25)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from domain.enums import WorkerStatus
from domain.events.base import DomainEvent, EventName
from domain.value_objects.identifiers import WorkerId

__all__ = [
    "WorkerDeregistered",
    "WorkerDraining",
    "WorkerHeartbeatReceived",
    "WorkerRegistered",
    "WorkerStatusChanged",
    "WorkerUnavailable",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerRegistered(DomainEvent):
    name: ClassVar[EventName] = "worker.registered"

    worker_id: WorkerId
    model_id: str
    endpoint: str
    max_concurrency: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerHeartbeatReceived(DomainEvent):
    name: ClassVar[EventName] = "worker.heartbeat"

    worker_id: WorkerId
    status: WorkerStatus
    active_jobs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerStatusChanged(DomainEvent):
    name: ClassVar[EventName] = "worker.status_changed"

    worker_id: WorkerId
    previous: WorkerStatus
    current: WorkerStatus


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerDraining(DomainEvent):
    name: ClassVar[EventName] = "worker.draining"

    worker_id: WorkerId
    active_jobs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerUnavailable(DomainEvent):
    """Emitted when heartbeats stopped: assigned jobs become retry candidates."""

    name: ClassVar[EventName] = "worker.unavailable"

    worker_id: WorkerId
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerDeregistered(DomainEvent):
    name: ClassVar[EventName] = "worker.deregistered"

    worker_id: WorkerId
    graceful: bool
