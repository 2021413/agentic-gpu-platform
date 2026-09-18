"""The run aggregate: one agentic objective against one project.

The run owns the workflow state, the retry budgets and the causal history of
what happened. Every transition goes through ``RunStateMachine``, so the set of
reachable states is small, enumerable and testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from domain.entities.base import Entity
from domain.enums import FailureKind, RunStatus
from domain.events.run import (
    CandidateSelected,
    PlanCompleted,
    PlanRequested,
    RepairRequested,
    ReviewRequested,
    RunCancelled,
    RunCompleted,
    RunCreated,
    RunFailed,
    RunStateChanged,
)
from domain.exceptions import RunNotModifiableError
from domain.services.run_state_machine import RunStateMachine
from domain.value_objects.identifiers import (
    CandidateId,
    IdempotencyKey,
    PlanId,
    ProjectId,
    RunId,
)
from domain.value_objects.limits import RunLimits
from domain.value_objects.llm import TokenUsage

__all__ = ["Run"]


class Run(Entity):
    """One agentic run. Resumable: all of its state is persisted, none is in RAM."""

    __slots__ = (
        "_candidate_count",
        "_completed_at",
        "_created_at",
        "_failure_kind",
        "_failure_reason",
        "_id",
        "_idempotency_key",
        "_limits",
        "_metadata",
        "_objective",
        "_plan_id",
        "_plan_revisions",
        "_project_id",
        "_repair_iterations",
        "_review_iterations",
        "_selected_candidate_id",
        "_started_at",
        "_status",
        "_token_usage",
        "_updated_at",
    )

    def __init__(
        self,
        *,
        run_id: RunId,
        project_id: ProjectId,
        objective: str,
        created_at: datetime,
        candidate_count: int = 1,
        limits: RunLimits | None = None,
        status: RunStatus = RunStatus.CREATED,
        plan_id: PlanId | None = None,
        plan_revisions: int = 0,
        repair_iterations: int = 0,
        review_iterations: int = 0,
        selected_candidate_id: CandidateId | None = None,
        token_usage: TokenUsage | None = None,
        idempotency_key: IdempotencyKey | None = None,
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        updated_at: datetime | None = None,
        failure_kind: FailureKind | None = None,
        failure_reason: str | None = None,
    ) -> None:
        super().__init__()
        if not objective.strip():
            raise ValueError("a run needs a non-empty objective")
        self._limits = limits or RunLimits()
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least 1")
        if candidate_count > self._limits.max_parallel_candidates:
            raise ValueError(
                f"candidate_count {candidate_count} exceeds the configured maximum "
                f"{self._limits.max_parallel_candidates}"
            )
        self._id = run_id
        self._project_id = project_id
        self._objective = objective.strip()
        self._candidate_count = candidate_count
        self._status = status
        self._plan_id = plan_id
        self._plan_revisions = plan_revisions
        self._repair_iterations = repair_iterations
        self._review_iterations = review_iterations
        self._selected_candidate_id = selected_candidate_id
        self._token_usage = token_usage or TokenUsage()
        self._idempotency_key = idempotency_key
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._created_at = created_at
        self._started_at = started_at
        self._completed_at = completed_at
        self._updated_at = updated_at or created_at
        self._failure_kind = failure_kind
        self._failure_reason = failure_reason

    # -- construction ---------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        run_id: RunId,
        project_id: ProjectId,
        objective: str,
        now: datetime,
        candidate_count: int = 1,
        limits: RunLimits | None = None,
        idempotency_key: IdempotencyKey | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Run:
        run = cls(
            run_id=run_id,
            project_id=project_id,
            objective=objective,
            created_at=now,
            candidate_count=candidate_count,
            limits=limits,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )
        run.record(
            RunCreated(
                occurred_at=now,
                run_id=run_id,
                project_id=project_id,
                objective=run.objective,
                candidate_count=candidate_count,
            )
        )
        return run

    # -- accessors ------------------------------------------------------
    @property
    def identity(self) -> RunId:
        return self._id

    @property
    def id(self) -> RunId:
        return self._id

    @property
    def project_id(self) -> ProjectId:
        return self._project_id

    @property
    def objective(self) -> str:
        return self._objective

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    @property
    def limits(self) -> RunLimits:
        return self._limits

    @property
    def plan_id(self) -> PlanId | None:
        return self._plan_id

    @property
    def plan_revisions(self) -> int:
        return self._plan_revisions

    @property
    def repair_iterations(self) -> int:
        return self._repair_iterations

    @property
    def review_iterations(self) -> int:
        return self._review_iterations

    @property
    def selected_candidate_id(self) -> CandidateId | None:
        return self._selected_candidate_id

    @property
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    @property
    def idempotency_key(self) -> IdempotencyKey | None:
        return self._idempotency_key

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def failure_kind(self) -> FailureKind | None:
        return self._failure_kind

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def is_terminal(self) -> bool:
        return self._status.is_terminal

    @property
    def is_cancelling(self) -> bool:
        return self._status.is_cancelling_or_cancelled

    @property
    def can_revise_plan(self) -> bool:
        return self._plan_revisions < self._limits.max_plan_revisions + 1

    @property
    def can_repair(self) -> bool:
        return self._repair_iterations < self._limits.max_repair_iterations

    # -- workflow -------------------------------------------------------
    def start_planning(self, now: datetime) -> None:
        self._transition(RunStatus.PLANNING, now)
        self._started_at = self._started_at or now
        self._plan_revisions += 1
        self.record(PlanRequested(occurred_at=now, run_id=self._id, revision=self._plan_revisions))

    def plan_ready(self, *, plan_id: PlanId, task_count: int, now: datetime) -> None:
        self._transition(RunStatus.PLAN_READY, now)
        self._plan_id = plan_id
        self.record(
            PlanCompleted(
                occurred_at=now,
                run_id=self._id,
                plan_id=plan_id,
                revision=self._plan_revisions,
                task_count=task_count,
            )
        )

    def start_coding(self, now: datetime) -> None:
        """Enter the coding stage, either after planning or from a repair loop."""
        self._transition(RunStatus.CODING, now)
        self._started_at = self._started_at or now

    def start_validating(self, now: datetime) -> None:
        self._transition(RunStatus.VALIDATING, now)

    def start_reviewing(self, *, candidate_id: CandidateId, now: datetime) -> None:
        self._transition(RunStatus.REVIEWING, now)
        self._review_iterations += 1
        self.record(
            ReviewRequested(
                occurred_at=now,
                run_id=self._id,
                candidate_id=candidate_id,
                iteration=self._review_iterations,
            )
        )

    def request_repair(self, *, candidate_id: CandidateId, reason: str, now: datetime) -> bool:
        """Enter a repair iteration. Returns False when the budget is exhausted.

        Returning a boolean rather than raising keeps the decision in the
        orchestrator: an exhausted budget is a normal outcome, not an error.
        """
        if not self.can_repair:
            return False
        self._transition(RunStatus.REPAIRING, now)
        self._repair_iterations += 1
        self.record(
            RepairRequested(
                occurred_at=now,
                run_id=self._id,
                candidate_id=candidate_id,
                iteration=self._repair_iterations,
                reason=reason,
            )
        )
        return True

    def request_plan_revision(self, now: datetime) -> bool:
        """Ask the planner for another revision if the budget allows it."""
        if not self.can_revise_plan:
            return False
        self._transition(RunStatus.PLANNING, now)
        self._plan_revisions += 1
        self.record(PlanRequested(occurred_at=now, run_id=self._id, revision=self._plan_revisions))
        return True

    def select_candidate(self, *, candidate_id: CandidateId, rationale: str, now: datetime) -> None:
        self._selected_candidate_id = candidate_id
        self._touch(now)
        self.record(
            CandidateSelected(
                occurred_at=now, run_id=self._id, candidate_id=candidate_id, rationale=rationale
            )
        )

    def complete(self, *, now: datetime, candidate_id: CandidateId | None = None) -> None:
        self._transition(RunStatus.COMPLETED, now)
        if candidate_id is not None:
            self._selected_candidate_id = candidate_id
        self._completed_at = now
        self.record(
            RunCompleted(occurred_at=now, run_id=self._id, candidate_id=self._selected_candidate_id)
        )

    def fail(self, *, now: datetime, kind: FailureKind, reason: str) -> None:
        if self._status.is_terminal:
            return
        self._transition(RunStatus.FAILED, now)
        self._failure_kind = kind
        self._failure_reason = reason
        self._completed_at = now
        self.record(RunFailed(occurred_at=now, run_id=self._id, failure_kind=kind, reason=reason))

    # -- cancellation (spec section 34) ----------------------------------
    def request_cancellation(self, *, now: datetime, reason: str | None = None) -> bool:
        """Mark the run as cancelling. Idempotent; returns False if already stopping."""
        if self._status.is_terminal or self._status is RunStatus.CANCELLING:
            return False
        self._transition(RunStatus.CANCELLING, now, reason=reason)
        return True

    def confirm_cancelled(self, *, now: datetime, reason: str | None = None) -> None:
        if self._status is RunStatus.CANCELLED:
            return
        self._transition(RunStatus.CANCELLED, now, reason=reason)
        self._failure_kind = FailureKind.CANCELLED
        self._completed_at = now
        self.record(RunCancelled(occurred_at=now, run_id=self._id, reason=reason))

    # -- accounting -----------------------------------------------------
    def add_token_usage(self, usage: TokenUsage, now: datetime) -> None:
        self._token_usage = self._token_usage + usage
        self._touch(now)

    def ensure_modifiable(self) -> None:
        if self._status.is_terminal:
            raise RunNotModifiableError(self._id, self._status)

    # -- internals ------------------------------------------------------
    def _transition(self, target: RunStatus, now: datetime, *, reason: str | None = None) -> None:
        previous = self._status
        RunStateMachine.ensure(previous, target)
        self._status = target
        self._touch(now)
        self.record(
            RunStateChanged(
                occurred_at=now,
                run_id=self._id,
                previous=previous,
                current=target,
                reason=reason,
            )
        )

    def _touch(self, now: datetime) -> None:
        self._updated_at = now

    def __repr__(self) -> str:
        return f"Run(id={self._id}, status={self._status}, candidates={self._candidate_count})"
