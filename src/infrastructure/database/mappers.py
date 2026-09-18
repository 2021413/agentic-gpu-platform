"""Translation between domain aggregates and ORM rows.

All knowledge of "how a ``ValidationReport`` looks in JSONB" lives here, in one
place, so that the domain never learns about columns and the ORM never learns
about invariants.

Two rules this module follows:

* Reconstitution goes through the aggregates' keyword-only constructors, which
  accept the full state precisely so that a row can be turned back into an
  object without replaying its history. ``create`` classmethods are never used
  here: they would emit a ``RunCreated`` event every time a run is read.
* Updating an existing row mutates the *same* model instance rather than
  merging a new one, because SQLAlchemy's optimistic version counter only works
  when the identity map sees the load and the change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan, PlanTask
from domain.entities.project import Project, ToolchainConfig
from domain.entities.review import Review, ReviewFinding, Severity
from domain.entities.run import Run
from domain.entities.worker import Worker
from domain.enums import (
    AgentRole,
    CandidateStatus,
    FailureKind,
    JobStatus,
    JobType,
    Priority,
    ReviewVerdict,
    RunStatus,
    WorkerStatus,
)
from domain.value_objects.identifiers import (
    CandidateId,
    IdempotencyKey,
    JobId,
    PlanId,
    ProjectId,
    ReviewId,
    RunId,
    TaskId,
    WorkerId,
    WorkspaceId,
)
from domain.value_objects.lease import Lease, LeaseToken
from domain.value_objects.limits import RunLimits
from domain.value_objects.llm import TokenUsage
from domain.value_objects.patch import FileChange, Patch
from domain.value_objects.tools import ToolKind, ToolResult
from domain.value_objects.validation import ValidationReport
from domain.value_objects.worker import (
    GpuSpec,
    JobRequirements,
    WorkerCapabilities,
    WorkerEndpoint,
    WorkerLoad,
)
from infrastructure.database.models import (
    CandidateModel,
    JobModel,
    PlanModel,
    PlanTaskModel,
    ProjectModel,
    ReviewFindingModel,
    ReviewModel,
    RunModel,
    ToolResultModel,
    WorkerModel,
)

__all__ = [
    "apply_candidate",
    "apply_job",
    "apply_run",
    "apply_worker",
    "candidate_to_domain",
    "candidate_to_model",
    "job_to_domain",
    "job_to_model",
    "plan_to_domain",
    "plan_to_model",
    "project_to_domain",
    "project_to_model",
    "review_to_domain",
    "review_to_model",
    "run_to_domain",
    "run_to_model",
    "tool_result_to_domain",
    "tool_result_to_model",
    "tool_results_to_models",
    "worker_to_domain",
    "worker_to_model",
]


# --------------------------------------------------------------------------
# project
# --------------------------------------------------------------------------
def project_to_model(project: Project) -> ProjectModel:
    return ProjectModel(
        id=project.id.value,
        name=project.name,
        repository_url=project.repository_url,
        local_path=project.local_path,
        default_branch=project.default_branch,
        created_at=project.created_at,
        toolchain=_dump_toolchain(project.toolchain),
        meta=dict(project.metadata),
    )


def project_to_domain(model: ProjectModel) -> Project:
    return Project(
        id=ProjectId(model.id),
        name=model.name,
        repository_url=model.repository_url,
        default_branch=model.default_branch,
        created_at=model.created_at,
        toolchain=_load_toolchain(model.toolchain),
        local_path=model.local_path,
        metadata=dict(model.meta),
    )


def _dump_toolchain(toolchain: ToolchainConfig) -> dict[str, Any]:
    return {
        "language": toolchain.language,
        "build_command": toolchain.build_command,
        "test_command": toolchain.test_command,
        "static_analysis_command": toolchain.static_analysis_command,
        "install_command": toolchain.install_command,
        "working_subdirectory": toolchain.working_subdirectory,
        "environment": dict(toolchain.environment),
    }


def _load_toolchain(raw: Mapping[str, Any]) -> ToolchainConfig:
    return ToolchainConfig(
        language=raw.get("language", "python"),
        build_command=raw.get("build_command"),
        test_command=raw.get("test_command"),
        static_analysis_command=raw.get("static_analysis_command"),
        install_command=raw.get("install_command"),
        working_subdirectory=raw.get("working_subdirectory"),
        environment=dict(raw.get("environment") or {}),
    )


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def run_to_model(run: Run) -> RunModel:
    model = RunModel(
        id=run.id.value,
        project_id=run.project_id.value,
        objective=run.objective,
        candidate_count=run.candidate_count,
        created_at=run.created_at,
        status=run.status.value,
        updated_at=run.updated_at,
    )
    apply_run(model, run)
    return model


def apply_run(model: RunModel, run: Run) -> None:
    """Copy the mutable state of a run onto its row."""
    model.status = run.status.value
    model.candidate_count = run.candidate_count
    model.limits = _dump_limits(run.limits)
    model.plan_id = run.plan_id.value if run.plan_id else None
    model.plan_revisions = run.plan_revisions
    model.repair_iterations = run.repair_iterations
    model.review_iterations = run.review_iterations
    model.selected_candidate_id = (
        run.selected_candidate_id.value if run.selected_candidate_id else None
    )
    model.input_tokens = run.token_usage.input_tokens
    model.output_tokens = run.token_usage.output_tokens
    model.idempotency_key = str(run.idempotency_key) if run.idempotency_key else None
    model.meta = dict(run.metadata)
    model.started_at = run.started_at
    model.completed_at = run.completed_at
    model.updated_at = run.updated_at
    model.failure_kind = run.failure_kind.value if run.failure_kind else None
    model.failure_reason = run.failure_reason


def run_to_domain(model: RunModel) -> Run:
    return Run(
        run_id=RunId(model.id),
        project_id=ProjectId(model.project_id),
        objective=model.objective,
        created_at=model.created_at,
        candidate_count=model.candidate_count,
        limits=_load_limits(model.limits),
        status=RunStatus(model.status),
        plan_id=PlanId(model.plan_id) if model.plan_id else None,
        plan_revisions=model.plan_revisions,
        repair_iterations=model.repair_iterations,
        review_iterations=model.review_iterations,
        selected_candidate_id=(
            CandidateId(model.selected_candidate_id) if model.selected_candidate_id else None
        ),
        token_usage=TokenUsage(input_tokens=model.input_tokens, output_tokens=model.output_tokens),
        idempotency_key=(IdempotencyKey(model.idempotency_key) if model.idempotency_key else None),
        metadata=dict(model.meta),
        started_at=model.started_at,
        completed_at=model.completed_at,
        updated_at=model.updated_at,
        failure_kind=FailureKind(model.failure_kind) if model.failure_kind else None,
        failure_reason=model.failure_reason,
    )


def _dump_limits(limits: RunLimits) -> dict[str, Any]:
    return {
        "max_plan_revisions": limits.max_plan_revisions,
        "max_coder_iterations": limits.max_coder_iterations,
        "max_repair_iterations": limits.max_repair_iterations,
        "max_worker_retries": limits.max_worker_retries,
        "max_parallel_candidates": limits.max_parallel_candidates,
    }


def _load_limits(raw: Mapping[str, Any]) -> RunLimits:
    defaults = RunLimits()
    return RunLimits(
        max_plan_revisions=raw.get("max_plan_revisions", defaults.max_plan_revisions),
        max_coder_iterations=raw.get("max_coder_iterations", defaults.max_coder_iterations),
        max_repair_iterations=raw.get("max_repair_iterations", defaults.max_repair_iterations),
        max_worker_retries=raw.get("max_worker_retries", defaults.max_worker_retries),
        max_parallel_candidates=raw.get(
            "max_parallel_candidates", defaults.max_parallel_candidates
        ),
    )


# --------------------------------------------------------------------------
# job
# --------------------------------------------------------------------------
def job_to_model(job: Job) -> JobModel:
    model = JobModel(
        id=job.id.value,
        run_id=job.run_id.value,
        project_id=job.project_id.value,
        type=job.type.value,
        priority=job.priority.value,
        status=job.status.value,
        created_at=job.created_at,
    )
    apply_job(model, job)
    return model


def apply_job(model: JobModel, job: Job) -> None:
    model.role = job.role.value if job.role else None
    model.candidate_id = job.candidate_id.value if job.candidate_id else None
    model.priority = job.priority.value
    model.status = job.status.value
    model.attempt = job.attempt
    model.max_attempts = job.max_attempts
    model.payload = dict(job.payload)
    model.requirements = _dump_requirements(job.requirements)
    model.idempotency_key = str(job.idempotency_key) if job.idempotency_key else None
    lease = job.lease
    # Flattened rather than nested: ``lease_expires_at`` is the indexed column
    # the reclaim scan reads.
    model.lease_token = str(lease.token) if lease else None
    model.lease_holder_id = lease.holder.value if lease else None
    model.lease_acquired_at = lease.acquired_at if lease else None
    model.lease_expires_at = lease.expires_at if lease else None
    model.assigned_worker_id = job.assigned_worker_id.value if job.assigned_worker_id else None
    model.started_at = job.started_at
    model.completed_at = job.completed_at
    model.result = dict(job.result) if job.result is not None else None
    model.failure_kind = job.failure_kind.value if job.failure_kind else None
    model.failure_reason = job.failure_reason


def job_to_domain(model: JobModel) -> Job:
    return Job(
        job_id=JobId(model.id),
        run_id=RunId(model.run_id),
        project_id=ProjectId(model.project_id),
        job_type=JobType(model.type),
        created_at=model.created_at,
        role=AgentRole(model.role) if model.role else None,
        candidate_id=CandidateId(model.candidate_id) if model.candidate_id else None,
        priority=Priority(model.priority),
        status=JobStatus(model.status),
        attempt=model.attempt,
        max_attempts=model.max_attempts,
        payload=dict(model.payload),
        requirements=_load_requirements(model.requirements),
        idempotency_key=(IdempotencyKey(model.idempotency_key) if model.idempotency_key else None),
        lease=_load_lease(model),
        assigned_worker_id=(
            WorkerId(model.assigned_worker_id) if model.assigned_worker_id else None
        ),
        started_at=model.started_at,
        completed_at=model.completed_at,
        result=dict(model.result) if model.result is not None else None,
        failure_kind=FailureKind(model.failure_kind) if model.failure_kind else None,
        failure_reason=model.failure_reason,
    )


def _load_lease(model: JobModel) -> Lease | None:
    if (
        model.lease_token is None
        or model.lease_holder_id is None
        or model.lease_acquired_at is None
        or model.lease_expires_at is None
    ):
        return None
    return Lease(
        job_id=JobId(model.id),
        holder=WorkerId(model.lease_holder_id),
        token=LeaseToken(model.lease_token),
        acquired_at=model.lease_acquired_at,
        expires_at=model.lease_expires_at,
    )


def _dump_requirements(requirements: JobRequirements | None) -> dict[str, Any] | None:
    if requirements is None:
        return None
    return {
        "role": requirements.role.value,
        "model_id": requirements.model_id,
        "estimated_prompt_tokens": requirements.estimated_prompt_tokens,
        "requires_tools": requirements.requires_tools,
        "requires_json_schema": requirements.requires_json_schema,
    }


def _load_requirements(raw: Mapping[str, Any] | None) -> JobRequirements | None:
    if not raw:
        return None
    return JobRequirements(
        role=AgentRole(raw["role"]),
        model_id=raw.get("model_id"),
        estimated_prompt_tokens=raw.get("estimated_prompt_tokens", 0),
        requires_tools=raw.get("requires_tools", False),
        requires_json_schema=raw.get("requires_json_schema", True),
    )


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def worker_to_model(worker: Worker) -> WorkerModel:
    model = WorkerModel(
        id=worker.id.value,
        endpoint=str(worker.endpoint),
        status=worker.status.value,
        registered_at=worker.registered_at,
        last_heartbeat_at=worker.last_heartbeat_at,
    )
    apply_worker(model, worker)
    return model


def apply_worker(model: WorkerModel, worker: Worker) -> None:
    model.endpoint = str(worker.endpoint)
    model.status = worker.status.value
    model.capabilities = _dump_capabilities(worker.capabilities)
    model.active_jobs = worker.load.active_jobs
    model.queued_jobs = worker.load.queued_jobs
    model.last_heartbeat_at = worker.last_heartbeat_at
    model.meta = dict(worker.metadata)


def worker_to_domain(model: WorkerModel) -> Worker:
    return Worker(
        worker_id=WorkerId(model.id),
        endpoint=WorkerEndpoint(model.endpoint),
        capabilities=_load_capabilities(model.capabilities),
        registered_at=model.registered_at,
        status=WorkerStatus(model.status),
        load=WorkerLoad(active_jobs=model.active_jobs, queued_jobs=model.queued_jobs),
        last_heartbeat_at=model.last_heartbeat_at,
        metadata=dict(model.meta),
    )


def _dump_capabilities(capabilities: WorkerCapabilities) -> dict[str, Any]:
    gpu = capabilities.gpu
    return {
        "model_id": capabilities.model_id,
        "context_length": capabilities.context_length,
        "max_concurrency": capabilities.max_concurrency,
        # Sorted so that two identical capability sets serialize identically,
        # which makes rows diffable and tests deterministic.
        "supported_roles": sorted(role.value for role in capabilities.supported_roles),
        "gpu": {
            "gpu_type": gpu.gpu_type,
            "gpu_count": gpu.gpu_count,
            "memory_gb": gpu.memory_gb,
            "tensor_parallel_size": gpu.tensor_parallel_size,
        },
        "supports_tools": capabilities.supports_tools,
        "supports_json_schema": capabilities.supports_json_schema,
        "metadata": dict(capabilities.metadata),
    }


def _load_capabilities(raw: Mapping[str, Any]) -> WorkerCapabilities:
    gpu = raw.get("gpu") or {}
    return WorkerCapabilities(
        model_id=raw["model_id"],
        context_length=raw["context_length"],
        max_concurrency=raw.get("max_concurrency", 1),
        supported_roles=frozenset(AgentRole(role) for role in raw.get("supported_roles", ())),
        gpu=GpuSpec(
            gpu_type=gpu.get("gpu_type"),
            gpu_count=gpu.get("gpu_count", 1),
            memory_gb=gpu.get("memory_gb"),
            tensor_parallel_size=gpu.get("tensor_parallel_size", 1),
        ),
        supports_tools=raw.get("supports_tools", True),
        supports_json_schema=raw.get("supports_json_schema", True),
        metadata=dict(raw.get("metadata") or {}),
    )


# --------------------------------------------------------------------------
# candidate
# --------------------------------------------------------------------------
def candidate_to_model(candidate: Candidate) -> CandidateModel:
    model = CandidateModel(
        id=candidate.id.value,
        run_id=candidate.run_id.value,
        candidate_index=candidate.index,
        status=candidate.status.value,
        created_at=candidate.created_at,
    )
    apply_candidate(model, candidate)
    return model


def apply_candidate(model: CandidateModel, candidate: Candidate) -> None:
    model.status = candidate.status.value
    model.workspace_id = candidate.workspace_id.value if candidate.workspace_id else None
    model.patch = _dump_patch(candidate.patch)
    model.validation = _dump_validation(candidate.validation)
    model.summary = candidate.summary
    model.uncertainties = list(candidate.uncertainties)
    model.coder_iterations = candidate.coder_iterations
    model.repair_iterations = candidate.repair_iterations
    model.worker_id = candidate.worker_id.value if candidate.worker_id else None
    model.last_job_id = candidate.last_job_id.value if candidate.last_job_id else None
    model.review_verdict = candidate.review_verdict.value if candidate.review_verdict else None
    model.started_at = candidate.started_at
    model.completed_at = candidate.completed_at
    model.failure_kind = candidate.failure_kind.value if candidate.failure_kind else None
    model.failure_reason = candidate.failure_reason


def candidate_to_domain(model: CandidateModel) -> Candidate:
    return Candidate(
        candidate_id=CandidateId(model.id),
        run_id=RunId(model.run_id),
        index=model.candidate_index,
        created_at=model.created_at,
        workspace_id=WorkspaceId(model.workspace_id) if model.workspace_id else None,
        status=CandidateStatus(model.status),
        patch=_load_patch(model.patch),
        validation=_load_validation(model.validation),
        summary=model.summary,
        uncertainties=tuple(model.uncertainties),
        coder_iterations=model.coder_iterations,
        repair_iterations=model.repair_iterations,
        worker_id=WorkerId(model.worker_id) if model.worker_id else None,
        last_job_id=JobId(model.last_job_id) if model.last_job_id else None,
        review_verdict=ReviewVerdict(model.review_verdict) if model.review_verdict else None,
        started_at=model.started_at,
        completed_at=model.completed_at,
        failure_kind=FailureKind(model.failure_kind) if model.failure_kind else None,
        failure_reason=model.failure_reason,
    )


def _dump_patch(patch: Patch | None) -> dict[str, Any] | None:
    if patch is None:
        return None
    return {
        "diff": patch.diff,
        "base_revision": patch.base_revision,
        # Stored rather than re-derived on read: the statistics of the patch as
        # it was produced are evidence, and a future parser fix must not
        # retroactively change what a past candidate was judged on.
        "files": [
            {
                "path": change.path,
                "added_lines": change.added_lines,
                "removed_lines": change.removed_lines,
            }
            for change in patch.files
        ],
    }


def _load_patch(raw: Mapping[str, Any] | None) -> Patch | None:
    if raw is None:
        return None
    return Patch(
        diff=raw.get("diff", ""),
        base_revision=raw.get("base_revision"),
        files=tuple(
            FileChange(
                path=change["path"],
                added_lines=change.get("added_lines", 0),
                removed_lines=change.get("removed_lines", 0),
            )
            for change in raw.get("files", ())
        ),
    )


def _dump_validation(report: ValidationReport) -> dict[str, Any]:
    return {
        "results": [_dump_tool_result(result) for result in report.results],
        "static_analysis_is_blocking": report.static_analysis_is_blocking,
    }


def _load_validation(raw: Mapping[str, Any]) -> ValidationReport:
    return ValidationReport(
        results=tuple(_load_tool_result(result) for result in raw.get("results", ())),
        static_analysis_is_blocking=raw.get("static_analysis_is_blocking", False),
    )


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def plan_to_model(plan: Plan) -> PlanModel:
    return PlanModel(
        id=plan.id.value,
        run_id=plan.run_id.value,
        revision=plan.revision,
        objective=plan.objective,
        created_at=plan.created_at,
        assumptions=list(plan.assumptions),
        constraints=list(plan.constraints),
        risk_areas=list(plan.risk_areas),
        validation_requirements=list(plan.validation_requirements),
        prompt_version=plan.prompt_version,
        meta=dict(plan.metadata),
        tasks=[
            PlanTaskModel(
                id=task.id.value,
                plan_id=plan.id.value,
                position=position,
                key=task.key,
                title=task.title,
                description=task.description,
                target_paths=list(task.target_paths),
                depends_on=list(task.depends_on),
                validation=list(task.validation),
            )
            for position, task in enumerate(plan.tasks)
        ],
    )


def plan_to_domain(model: PlanModel) -> Plan:
    return Plan(
        id=PlanId(model.id),
        run_id=RunId(model.run_id),
        revision=model.revision,
        objective=model.objective,
        created_at=model.created_at,
        tasks=tuple(
            PlanTask(
                id=TaskId(task.id),
                key=task.key,
                title=task.title,
                description=task.description,
                target_paths=tuple(task.target_paths),
                depends_on=tuple(task.depends_on),
                validation=tuple(task.validation),
            )
            for task in sorted(model.tasks, key=lambda task: task.position)
        ),
        assumptions=tuple(model.assumptions),
        constraints=tuple(model.constraints),
        risk_areas=tuple(model.risk_areas),
        validation_requirements=tuple(model.validation_requirements),
        prompt_version=model.prompt_version,
        metadata=dict(model.meta),
    )


# --------------------------------------------------------------------------
# review
# --------------------------------------------------------------------------
def review_to_model(review: Review) -> ReviewModel:
    return ReviewModel(
        id=review.id.value,
        run_id=review.run_id.value,
        candidate_id=review.candidate_id.value,
        verdict=review.verdict.value,
        iteration=review.iteration,
        created_at=review.created_at,
        summary=review.summary,
        prompt_version=review.prompt_version,
        meta=dict(review.metadata),
        findings=[
            ReviewFindingModel(
                # Findings have no domain identity; the row needs one anyway.
                id=uuid4(),
                review_id=review.id.value,
                position=position,
                summary=finding.summary,
                severity=finding.severity.value,
                file_path=finding.file,
                line=finding.line,
                repair_instruction=finding.repair_instruction,
            )
            for position, finding in enumerate(review.findings)
        ],
    )


def review_to_domain(model: ReviewModel) -> Review:
    return Review(
        id=ReviewId(model.id),
        run_id=RunId(model.run_id),
        candidate_id=CandidateId(model.candidate_id),
        verdict=ReviewVerdict(model.verdict),
        iteration=model.iteration,
        created_at=model.created_at,
        summary=model.summary,
        findings=tuple(
            ReviewFinding(
                summary=finding.summary,
                severity=Severity(finding.severity),
                file=finding.file_path,
                line=finding.line,
                repair_instruction=finding.repair_instruction,
            )
            for finding in sorted(model.findings, key=lambda finding: finding.position)
        ),
        prompt_version=model.prompt_version,
        metadata=dict(model.meta),
    )


# --------------------------------------------------------------------------
# tool results
# --------------------------------------------------------------------------
def tool_result_to_model(
    result: ToolResult,
    *,
    run_id: RunId,
    candidate_id: CandidateId | None,
    position: int,
    row_id: UUID | None = None,
) -> ToolResultModel:
    """Row for one tool execution.

    ``created_at`` is left unset on purpose so PostgreSQL stamps it: the domain
    value object carries no storage timestamp and inventing one here would be
    reading an ambient clock.
    """
    return ToolResultModel(
        id=row_id or uuid4(),
        run_id=run_id.value,
        candidate_id=candidate_id.value if candidate_id else None,
        position=position,
        tool=result.tool,
        kind=result.kind.value,
        command=result.command,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_ms=result.duration_ms,
        truncated=result.truncated,
        artifacts=list(result.artifacts),
        meta=dict(result.metadata),
        duration_seconds=result.duration_ms / 1000,
    )


def tool_result_to_domain(model: ToolResultModel) -> ToolResult:
    return ToolResult(
        tool=model.tool,
        kind=ToolKind(model.kind),
        command=model.command,
        exit_code=model.exit_code,
        stdout=model.stdout,
        stderr=model.stderr,
        duration_ms=model.duration_ms,
        truncated=model.truncated,
        artifacts=tuple(model.artifacts),
        metadata=dict(model.meta),
    )


def _dump_tool_result(result: ToolResult) -> dict[str, Any]:
    return {
        "tool": result.tool,
        "kind": result.kind.value,
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
        "truncated": result.truncated,
        "artifacts": list(result.artifacts),
        "metadata": dict(result.metadata),
    }


def _load_tool_result(raw: Mapping[str, Any]) -> ToolResult:
    return ToolResult(
        tool=raw["tool"],
        kind=ToolKind(raw["kind"]),
        command=raw.get("command", ""),
        exit_code=raw.get("exit_code", 0),
        stdout=raw.get("stdout", ""),
        stderr=raw.get("stderr", ""),
        duration_ms=raw.get("duration_ms", 0),
        truncated=raw.get("truncated", False),
        artifacts=tuple(raw.get("artifacts", ())),
        metadata=dict(raw.get("metadata") or {}),
    )


def tool_results_to_models(
    results: Sequence[ToolResult],
    *,
    run_id: RunId,
    candidate_id: CandidateId | None,
    first_position: int,
) -> list[ToolResultModel]:
    """Rows for a batch of results, numbered from ``first_position``."""
    return [
        tool_result_to_model(
            result,
            run_id=run_id,
            candidate_id=candidate_id,
            position=first_position + offset,
        )
        for offset, result in enumerate(results)
    ]
