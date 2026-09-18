"""Domain events.

Importing from this package gives the whole vocabulary; the submodules group
them by aggregate.
"""

from __future__ import annotations

from domain.events.base import DomainEvent, EventName
from domain.events.job import (
    JobCompleted,
    JobEnqueued,
    JobFailed,
    JobLeased,
    JobLeaseExpired,
    JobRequeued,
)
from domain.events.run import (
    CandidateCompleted,
    CandidateSelected,
    CandidateStarted,
    PlanCompleted,
    PlanRequested,
    RepairRequested,
    ReviewCompleted,
    ReviewRequested,
    RunCancelled,
    RunCompleted,
    RunCreated,
    RunFailed,
    RunStateChanged,
    ValidationCompleted,
    ValidationStarted,
)
from domain.events.worker import (
    WorkerDeregistered,
    WorkerDraining,
    WorkerHeartbeatReceived,
    WorkerRegistered,
    WorkerStatusChanged,
    WorkerUnavailable,
)

__all__ = [
    "CandidateCompleted",
    "CandidateSelected",
    "CandidateStarted",
    "DomainEvent",
    "EventName",
    "JobCompleted",
    "JobEnqueued",
    "JobFailed",
    "JobLeaseExpired",
    "JobLeased",
    "JobRequeued",
    "PlanCompleted",
    "PlanRequested",
    "RepairRequested",
    "ReviewCompleted",
    "ReviewRequested",
    "RunCancelled",
    "RunCompleted",
    "RunCreated",
    "RunFailed",
    "RunStateChanged",
    "ValidationCompleted",
    "ValidationStarted",
    "WorkerDeregistered",
    "WorkerDraining",
    "WorkerHeartbeatReceived",
    "WorkerRegistered",
    "WorkerStatusChanged",
    "WorkerUnavailable",
]
