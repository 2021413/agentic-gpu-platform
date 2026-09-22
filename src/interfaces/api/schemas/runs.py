"""Run, plan, candidate and event schemas (spec section 46).

Identifiers are UUIDs on the wire; the typed identifiers of the domain
(``RunId``, ``CandidateId``, ...) never cross the boundary, and neither do
aggregates — only the flat views the application returns.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from application.dto.commands import CancelRunCommand, CreateRunCommand
from application.dto.views import (
    CandidatePatchView,
    CandidateView,
    EventView,
    PlanView,
    ReviewFindingView,
    ReviewView,
    RunDetailView,
    RunView,
)
from domain.enums import CandidateStatus, FailureKind, ReviewVerdict, RunStatus
from domain.value_objects.identifiers import IdempotencyKey, ProjectId, RunId

__all__ = [
    "CancelRunRequest",
    "CandidateListResponse",
    "CandidatePatchResponse",
    "CandidateResponse",
    "CreateRunRequest",
    "PlanResponse",
    "PlanTaskResponse",
    "RejectRunRequest",
    "ReviewFindingResponse",
    "ReviewListResponse",
    "ReviewResponse",
    "RunDetailResponse",
    "RunEventResponse",
    "RunListResponse",
    "RunResponse",
]

MAX_OBJECTIVE_LENGTH = 20_000
"""An objective is a brief, not a codebase: anything longer is a paste accident.

Bounded here because the objective ends up inside every planner prompt, where an
unbounded string silently becomes a context-window failure much later.
"""

MAX_CANDIDATE_COUNT = 16
"""Upper bound accepted at the boundary; the run's own ``RunLimits`` clamps
further. The API refuses absurd values early so the platform never allocates
work it would immediately discard.
"""


class CreateRunRequest(BaseModel):
    """Start an agentic run against a project."""

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=MAX_OBJECTIVE_LENGTH)
    candidate_count: int | None = Field(
        default=None,
        ge=1,
        le=MAX_CANDIDATE_COUNT,
        description=(
            "How many implementations to race. Omit it to let the task-complexity policy decide."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_command(self, *, project_id: UUID, idempotency_key: str | None) -> CreateRunCommand:
        """Translate into the application command.

        The idempotency key travels in the ``Idempotency-Key`` header rather
        than in the body: it qualifies the *delivery* of the request, not the
        run being described, and a retry must be able to resend an identical
        body unchanged.
        """
        return CreateRunCommand(
            project_id=ProjectId(project_id),
            objective=self.objective,
            candidate_count=self.candidate_count,
            idempotency_key=IdempotencyKey(idempotency_key) if idempotency_key else None,
            metadata=dict(self.metadata),
        )


class CancelRunRequest(BaseModel):
    """Optional body of a cancellation; the reason is stored with the run."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=1000)

    def to_command(self, *, run_id: UUID) -> CancelRunCommand:
        return CancelRunCommand(run_id=RunId(run_id), reason=self.reason)


class RunResponse(BaseModel):
    """State of a run, as the client sees it."""

    id: UUID
    project_id: UUID
    status: RunStatus
    objective: str
    candidate_count: int
    plan_revisions: int
    repair_iterations: int
    selected_candidate_id: UUID | None
    input_tokens: int
    output_tokens: int
    failure_kind: FailureKind | None
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @classmethod
    def of(cls, view: RunView) -> RunResponse:
        return cls(
            id=view.id.value,
            project_id=view.project_id.value,
            status=view.status,
            objective=view.objective,
            candidate_count=view.candidate_count,
            plan_revisions=view.plan_revisions,
            repair_iterations=view.repair_iterations,
            selected_candidate_id=(
                view.selected_candidate_id.value if view.selected_candidate_id else None
            ),
            input_tokens=view.input_tokens,
            output_tokens=view.output_tokens,
            failure_kind=view.failure_kind,
            failure_reason=view.failure_reason,
            created_at=view.created_at,
            updated_at=view.updated_at,
            completed_at=view.completed_at,
        )


class PlanTaskResponse(BaseModel):
    key: str
    title: str
    description: str
    depends_on: list[str]
    target_paths: list[str]


class PlanResponse(BaseModel):
    """The planner's output: tasks and their dependencies, never its reasoning."""

    id: UUID
    run_id: UUID
    revision: int
    objective: str
    tasks: list[PlanTaskResponse]
    assumptions: list[str]
    constraints: list[str]
    risk_areas: list[str]
    max_parallelism: int
    created_at: datetime

    @classmethod
    def of(cls, view: PlanView) -> PlanResponse:
        return cls(
            id=view.id.value,
            run_id=view.run_id.value,
            revision=view.revision,
            objective=view.objective,
            tasks=[
                PlanTaskResponse(
                    key=task.key,
                    title=task.title,
                    description=task.description,
                    depends_on=list(task.depends_on),
                    target_paths=list(task.target_paths),
                )
                for task in view.tasks
            ],
            assumptions=list(view.assumptions),
            constraints=list(view.constraints),
            risk_areas=list(view.risk_areas),
            max_parallelism=view.max_parallelism,
            created_at=view.created_at,
        )


class CandidateResponse(BaseModel):
    """One competing implementation attempt and the facts about it."""

    id: UUID
    run_id: UUID
    index: int
    status: CandidateStatus
    viable: bool
    build_passed: bool | None
    tests_passed: bool | None
    validation_summary: str
    changed_files: list[str]
    total_churn: int
    review_verdict: ReviewVerdict | None
    coder_iterations: int
    repair_iterations: int
    worker_id: UUID | None
    summary: str
    uncertainties: list[str]

    @classmethod
    def of(cls, view: CandidateView) -> CandidateResponse:
        return cls(
            id=view.id.value,
            run_id=view.run_id.value,
            index=view.index,
            status=view.status,
            viable=view.viable,
            build_passed=view.build_passed,
            tests_passed=view.tests_passed,
            validation_summary=view.validation_summary,
            changed_files=list(view.changed_files),
            total_churn=view.total_churn,
            review_verdict=view.review_verdict,
            coder_iterations=view.coder_iterations,
            repair_iterations=view.repair_iterations,
            worker_id=view.worker_id.value if view.worker_id else None,
            summary=view.summary,
            uncertainties=list(view.uncertainties),
        )


class RunDetailResponse(BaseModel):
    """A run plus, when asked for, its latest plan and its candidates."""

    run: RunResponse
    plan: PlanResponse | None = None
    candidates: list[CandidateResponse] = Field(default_factory=list)

    @classmethod
    def of(cls, view: RunDetailView) -> RunDetailResponse:
        return cls(
            run=RunResponse.of(view.run),
            plan=PlanResponse.of(view.plan) if view.plan is not None else None,
            candidates=[CandidateResponse.of(candidate) for candidate in view.candidates],
        )


class RunEventResponse(BaseModel):
    """One entry of a run's event stream.

    ``sequence`` is the resumption cursor: it is present for events replayed
    from the durable store and ``null`` for events delivered live, which have
    not been numbered yet.
    """

    sequence: int | None
    name: str
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, view: EventView, *, payload: dict[str, Any]) -> RunEventResponse:
        return cls(
            sequence=view.sequence,
            name=view.name,
            occurred_at=view.occurred_at,
            payload=payload,
        )


class RejectRunRequest(BaseModel):
    """Why the patch was refused.

    Required, and not cosmetic: it is what the coder is handed on the next
    round. An empty reason spends a repair to learn nothing.
    """

    reason: str = Field(min_length=1, max_length=2000)


class RunListResponse(BaseModel):
    """Runs of one project, newest first."""

    runs: list[RunResponse]

    @classmethod
    def of(cls, views: Sequence[RunView]) -> RunListResponse:
        return cls(runs=[RunResponse.of(v) for v in views])


class CandidatePatchResponse(BaseModel):
    """The code a candidate wrote, in full."""

    candidate_id: UUID
    run_id: UUID
    index: int
    diff: str
    changed_files: list[str]
    total_churn: int
    base_revision: str | None

    @classmethod
    def of(cls, view: CandidatePatchView) -> CandidatePatchResponse:
        return cls(
            candidate_id=view.candidate_id.value,
            run_id=view.run_id.value,
            index=view.index,
            diff=view.diff,
            changed_files=list(view.changed_files),
            total_churn=view.total_churn,
            base_revision=view.base_revision,
        )


class ReviewFindingResponse(BaseModel):
    summary: str
    severity: str
    file: str | None
    line: int | None
    repair_instruction: str | None

    @classmethod
    def of(cls, view: ReviewFindingView) -> ReviewFindingResponse:
        return cls(
            summary=view.summary,
            severity=str(view.severity),
            file=view.file,
            line=view.line,
            repair_instruction=view.repair_instruction,
        )


class ReviewResponse(BaseModel):
    """One reviewer verdict. Append-only: a repair loop adds, never replaces."""

    id: UUID
    run_id: UUID
    candidate_id: UUID
    verdict: str
    iteration: int
    summary: str
    findings: list[ReviewFindingResponse]
    created_at: datetime

    @classmethod
    def of(cls, view: ReviewView) -> ReviewResponse:
        return cls(
            id=view.id.value,
            run_id=view.run_id.value,
            candidate_id=view.candidate_id.value,
            verdict=str(view.verdict),
            iteration=view.iteration,
            summary=view.summary,
            findings=[ReviewFindingResponse.of(f) for f in view.findings],
            created_at=view.created_at,
        )


class ReviewListResponse(BaseModel):
    reviews: list[ReviewResponse]

    @classmethod
    def of(cls, views: Sequence[ReviewView]) -> ReviewListResponse:
        return cls(reviews=[ReviewResponse.of(v) for v in views])


class CandidateListResponse(BaseModel):
    candidates: list[CandidateResponse]

    @classmethod
    def of(cls, views: Sequence[CandidateView]) -> CandidateListResponse:
        return cls(candidates=[CandidateResponse.of(view) for view in views])
