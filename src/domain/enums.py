"""Domain enumerations.

These are part of the ubiquitous language: persistence and the HTTP API both
serialize them by ``value``, so the string values are a compatibility contract.
"""

from __future__ import annotations

from enum import StrEnum, unique

__all__ = [
    "AgentRole",
    "CandidateStatus",
    "FailureKind",
    "JobStatus",
    "JobType",
    "Priority",
    "ReviewVerdict",
    "RunStatus",
    "WorkerStatus",
]


@unique
class RunStatus(StrEnum):
    """Lifecycle of an agentic run. Transitions are governed by ``RunStateMachine``."""

    CREATED = "CREATED"
    PLANNING = "PLANNING"
    PLAN_READY = "PLAN_READY"
    CODING = "CODING"
    VALIDATING = "VALIDATING"
    REVIEWING = "REVIEWING"
    REPAIRING = "REPAIRING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    """Reviewed and waiting for a human to let it land.

    Only reachable when the deployment asks for it. Integration is a write into
    someone else's repository, and a run that waits for an approval nobody is
    watching never finishes — so the gate is opt-in, and this state is not
    terminal: it must keep appearing in the active list to be approvable at all.
    """
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_RUN_STATUSES

    @property
    def is_cancelling_or_cancelled(self) -> bool:
        return self in (RunStatus.CANCELLING, RunStatus.CANCELLED)


_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED},
)


@unique
class JobStatus(StrEnum):
    """Lifecycle of a job in the distributed queue."""

    PENDING = "PENDING"
    """Created and persisted, not yet published to the queue."""

    QUEUED = "QUEUED"
    """Available for a consumer to claim."""

    LEASED = "LEASED"
    """Claimed by a consumer holding a time-bounded lease."""

    RUNNING = "RUNNING"
    """Actively executing; the lease is being renewed."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    """Failed but still eligible for another attempt."""

    DEAD = "DEAD"
    """Failed and out of attempts."""

    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_JOB_STATUSES

    @property
    def is_in_flight(self) -> bool:
        return self in (JobStatus.LEASED, JobStatus.RUNNING)


_TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.DEAD, JobStatus.CANCELLED},
)


@unique
class JobType(StrEnum):
    """What a job actually does."""

    PLAN = "PLAN"
    CODE = "CODE"
    REVIEW = "REVIEW"
    LLM_INFERENCE = "LLM_INFERENCE"
    BUILD = "BUILD"
    TEST = "TEST"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    PATCH_APPLY = "PATCH_APPLY"

    @property
    def requires_inference(self) -> bool:
        """Inference jobs need a compatible GPU worker; deterministic jobs do not."""
        return self in _INFERENCE_JOB_TYPES


_INFERENCE_JOB_TYPES = frozenset(
    {JobType.PLAN, JobType.CODE, JobType.REVIEW, JobType.LLM_INFERENCE},
)


@unique
class AgentRole(StrEnum):
    """Logical LLM role. Several roles may share one underlying model."""

    PLANNER = "PLANNER"
    CODER = "CODER"
    REVIEWER = "REVIEWER"


@unique
class WorkerStatus(StrEnum):
    """GPU worker lifecycle (spec section 25)."""

    STARTING = "STARTING"
    REGISTERING = "REGISTERING"
    READY = "READY"
    BUSY = "BUSY"
    DRAINING = "DRAINING"
    OFFLINE = "OFFLINE"
    UNHEALTHY = "UNHEALTHY"

    @property
    def accepts_new_jobs(self) -> bool:
        """``BUSY`` still accepts work as long as capacity remains; ``DRAINING`` never does."""
        return self in (WorkerStatus.READY, WorkerStatus.BUSY)

    @property
    def is_live(self) -> bool:
        """A live worker may still be finishing already-accepted jobs."""
        return self not in (WorkerStatus.OFFLINE, WorkerStatus.UNHEALTHY)


@unique
class CandidateStatus(StrEnum):
    """Lifecycle of one competing implementation attempt."""

    CREATED = "CREATED"
    CODING = "CODING"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    """Deterministic validation finished; results decide viability."""

    REJECTED = "REJECTED"
    SELECTED = "SELECTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            CandidateStatus.REJECTED,
            CandidateStatus.SELECTED,
            CandidateStatus.FAILED,
            CandidateStatus.CANCELLED,
        )


@unique
class ReviewVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@unique
class Priority(StrEnum):
    """Scheduling priority. Ordering is defined by ``Priority.rank``."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Higher rank is dequeued first."""
        return _PRIORITY_RANKS[self]


_PRIORITY_RANKS: dict[Priority, int] = {
    Priority.LOW: 0,
    Priority.NORMAL: 10,
    Priority.HIGH: 20,
    Priority.CRITICAL: 30,
}


@unique
class FailureKind(StrEnum):
    """Why something failed.

    The retry policy (spec section 11) branches on this: infrastructure problems
    are retried on another worker, code problems go back into the agentic repair
    loop. Never treat every failure identically.
    """

    INFRASTRUCTURE = "INFRASTRUCTURE"
    """Worker vanished, lease expired, connection refused, queue error."""

    INFERENCE = "INFERENCE"
    """The model call itself failed: timeout, context overflow, server error."""

    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    """The model answered, but the answer did not validate against the schema."""

    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    """The answer ran out of room before it was finished.

    Distinct from INVALID_STRUCTURED_OUTPUT although it arrives looking like
    it: a truncated answer is malformed, but re-asking produces the identical
    truncation. It is a budget problem and belongs to whoever sets the window.
    """

    TOOL = "TOOL"
    """A deterministic tool could not be executed at all."""

    COMPILATION = "COMPILATION"
    TEST = "TEST"
    REVIEW = "REVIEW"
    CANCELLED = "CANCELLED"

    @property
    def is_infrastructure(self) -> bool:
        """Failures of the machinery rather than of the produced code."""
        return self in (FailureKind.INFRASTRUCTURE, FailureKind.INFERENCE, FailureKind.TOOL)

    @property
    def is_code_defect(self) -> bool:
        """Failures that the agentic repair loop can plausibly fix."""
        return self in (FailureKind.COMPILATION, FailureKind.TEST, FailureKind.REVIEW)
