"""Read models returned by queries.

Views are flat and serialization-friendly so the API layer only has to rename
fields, never to reach into an aggregate. Keeping aggregates out of responses is
what stops HTTP concerns from leaking inwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from domain.entities.candidate import Candidate
from domain.entities.plan import Plan
from domain.entities.project import Project, ToolchainConfig
from domain.entities.run import Run
from domain.entities.worker import Worker
from domain.enums import CandidateStatus, FailureKind, ReviewVerdict, RunStatus, WorkerStatus
from domain.value_objects.identifiers import CandidateId, PlanId, ProjectId, RunId, WorkerId

__all__ = [
    "CandidateView",
    "EventView",
    "PlanTaskView",
    "PlanView",
    "ProjectView",
    "RunView",
    "WorkerView",
]


@dataclass(frozen=True, slots=True)
class ProjectView:
    id: ProjectId
    name: str
    repository_url: str | None
    default_branch: str
    language: str
    created_at: datetime
    toolchain: ToolchainConfig = field(default_factory=ToolchainConfig)
    """Carried whole, not just its language.

    A project runs these commands against the caller's code. They were stored
    and executed but never shown, so there was no way to answer "what is this
    about to run", nor to notice that a project created earlier kept commands
    that have since changed — a project's toolchain is fixed at creation.
    """

    @classmethod
    def of(cls, project: Project) -> ProjectView:
        return cls(
            id=project.id,
            name=project.name,
            repository_url=project.repository_url,
            default_branch=project.default_branch,
            language=project.toolchain.language,
            created_at=project.created_at,
            toolchain=project.toolchain,
        )


@dataclass(frozen=True, slots=True)
class RunView:
    id: RunId
    project_id: ProjectId
    status: RunStatus
    objective: str
    candidate_count: int
    plan_revisions: int
    repair_iterations: int
    selected_candidate_id: CandidateId | None
    input_tokens: int
    output_tokens: int
    failure_kind: FailureKind | None
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @classmethod
    def of(cls, run: Run) -> RunView:
        return cls(
            id=run.id,
            project_id=run.project_id,
            status=run.status,
            objective=run.objective,
            candidate_count=run.candidate_count,
            plan_revisions=run.plan_revisions,
            repair_iterations=run.repair_iterations,
            selected_candidate_id=run.selected_candidate_id,
            input_tokens=run.token_usage.input_tokens,
            output_tokens=run.token_usage.output_tokens,
            failure_kind=run.failure_kind,
            failure_reason=run.failure_reason,
            created_at=run.created_at,
            updated_at=run.updated_at,
            completed_at=run.completed_at,
        )


@dataclass(frozen=True, slots=True)
class CandidateView:
    id: CandidateId
    run_id: RunId
    index: int
    status: CandidateStatus
    viable: bool
    build_passed: bool | None
    tests_passed: bool | None
    validation_summary: str
    changed_files: tuple[str, ...]
    total_churn: int
    review_verdict: ReviewVerdict | None
    coder_iterations: int
    repair_iterations: int
    worker_id: WorkerId | None
    summary: str
    uncertainties: tuple[str, ...]

    @classmethod
    def of(cls, candidate: Candidate) -> CandidateView:
        patch = candidate.patch
        validation = candidate.validation
        return cls(
            id=candidate.id,
            run_id=candidate.run_id,
            index=candidate.index,
            status=candidate.status,
            viable=candidate.is_viable,
            build_passed=validation.build_passed,
            tests_passed=validation.tests_passed,
            validation_summary=validation.summary(),
            changed_files=patch.changed_paths if patch else (),
            total_churn=patch.total_churn if patch else 0,
            review_verdict=candidate.review_verdict,
            coder_iterations=candidate.coder_iterations,
            repair_iterations=candidate.repair_iterations,
            worker_id=candidate.worker_id,
            summary=candidate.summary,
            uncertainties=candidate.uncertainties,
        )


@dataclass(frozen=True, slots=True)
class PlanTaskView:
    key: str
    title: str
    description: str
    depends_on: tuple[str, ...]
    target_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanView:
    id: PlanId
    run_id: RunId
    revision: int
    objective: str
    tasks: tuple[PlanTaskView, ...]
    assumptions: tuple[str, ...]
    constraints: tuple[str, ...]
    risk_areas: tuple[str, ...]
    max_parallelism: int
    created_at: datetime

    @classmethod
    def of(cls, plan: Plan) -> PlanView:
        return cls(
            id=plan.id,
            run_id=plan.run_id,
            revision=plan.revision,
            objective=plan.objective,
            tasks=tuple(
                PlanTaskView(
                    key=t.key,
                    title=t.title,
                    description=t.description,
                    depends_on=t.depends_on,
                    target_paths=t.target_paths,
                )
                for t in plan.tasks
            ),
            assumptions=plan.assumptions,
            constraints=plan.constraints,
            risk_areas=plan.risk_areas,
            max_parallelism=plan.max_parallelism,
            created_at=plan.created_at,
        )


@dataclass(frozen=True, slots=True)
class WorkerView:
    id: WorkerId
    model_id: str
    status: WorkerStatus
    endpoint: str
    capacity: int
    active_jobs: int
    context_length: int
    gpu_type: str | None
    gpu_count: int
    supported_roles: tuple[str, ...]
    registered_at: datetime
    last_heartbeat_at: datetime

    @classmethod
    def of(cls, worker: Worker) -> WorkerView:
        caps = worker.capabilities
        return cls(
            id=worker.id,
            model_id=caps.model_id,
            status=worker.status,
            endpoint=str(worker.endpoint),
            capacity=caps.max_concurrency,
            active_jobs=worker.active_jobs,
            context_length=caps.context_length,
            gpu_type=caps.gpu.gpu_type,
            gpu_count=caps.gpu.gpu_count,
            supported_roles=tuple(sorted(str(r) for r in caps.supported_roles)),
            registered_at=worker.registered_at,
            last_heartbeat_at=worker.last_heartbeat_at,
        )


@dataclass(frozen=True, slots=True)
class EventView:
    """One entry of a run's event stream, addressable for SSE resumption."""

    sequence: int
    name: str
    occurred_at: datetime
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunDetailView:
    """Everything a client needs to render a run in one response."""

    run: RunView
    plan: PlanView | None = None
    candidates: Sequence[CandidateView] = ()
