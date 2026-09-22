"""Events describing the life of a run, its candidates and its reviews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from domain.enums import CandidateStatus, FailureKind, ReviewVerdict, RunStatus
from domain.events.base import DomainEvent, EventName
from domain.value_objects.identifiers import CandidateId, JobId, PlanId, ProjectId, RunId

__all__ = [
    "CandidateCompleted",
    "CandidateSelected",
    "CandidateStarted",
    "PlanCompleted",
    "PlanRequested",
    "RepairRequested",
    "ReviewCompleted",
    "ReviewRequested",
    "RunApprovalRejected",
    "RunAwaitingApproval",
    "RunCancelled",
    "RunCompleted",
    "RunCreated",
    "RunFailed",
    "RunStateChanged",
    "ValidationCompleted",
    "ValidationStarted",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCreated(DomainEvent):
    name: ClassVar[EventName] = "run.created"

    run_id: RunId
    project_id: ProjectId
    objective: str
    candidate_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStateChanged(DomainEvent):
    name: ClassVar[EventName] = "run.state_changed"

    run_id: RunId
    previous: RunStatus
    current: RunStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanRequested(DomainEvent):
    name: ClassVar[EventName] = "run.plan_requested"

    run_id: RunId
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanCompleted(DomainEvent):
    name: ClassVar[EventName] = "run.plan_completed"

    run_id: RunId
    plan_id: PlanId
    revision: int
    task_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateStarted(DomainEvent):
    name: ClassVar[EventName] = "candidate.started"

    run_id: RunId
    candidate_id: CandidateId
    index: int
    job_id: JobId | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateCompleted(DomainEvent):
    name: ClassVar[EventName] = "candidate.completed"

    run_id: RunId
    candidate_id: CandidateId
    status: CandidateStatus
    changed_files: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationStarted(DomainEvent):
    name: ClassVar[EventName] = "candidate.validation_started"

    run_id: RunId
    candidate_id: CandidateId


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationCompleted(DomainEvent):
    name: ClassVar[EventName] = "candidate.validation_completed"

    run_id: RunId
    candidate_id: CandidateId
    viable: bool
    summary: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewRequested(DomainEvent):
    name: ClassVar[EventName] = "review.requested"

    run_id: RunId
    candidate_id: CandidateId
    iteration: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewCompleted(DomainEvent):
    name: ClassVar[EventName] = "review.completed"

    run_id: RunId
    candidate_id: CandidateId
    verdict: ReviewVerdict
    finding_count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class RepairRequested(DomainEvent):
    name: ClassVar[EventName] = "run.repair_requested"

    run_id: RunId
    candidate_id: CandidateId
    iteration: int
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateSelected(DomainEvent):
    name: ClassVar[EventName] = "candidate.selected"

    run_id: RunId
    candidate_id: CandidateId
    rationale: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(DomainEvent):
    name: ClassVar[EventName] = "run.completed"

    run_id: RunId
    candidate_id: CandidateId | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunAwaitingApproval(DomainEvent):
    """Reviewed, and waiting for a human to let it land."""

    name: ClassVar[EventName] = "run.awaiting_approval"

    run_id: RunId
    candidate_id: CandidateId | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunApprovalRejected(DomainEvent):
    """A human refused the patch. The reason is what the coder gets to act on."""

    name: ClassVar[EventName] = "run.approval_rejected"

    run_id: RunId
    reason: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(DomainEvent):
    name: ClassVar[EventName] = "run.failed"

    run_id: RunId
    failure_kind: FailureKind
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCancelled(DomainEvent):
    name: ClassVar[EventName] = "run.cancelled"

    run_id: RunId
    reason: str | None = None
