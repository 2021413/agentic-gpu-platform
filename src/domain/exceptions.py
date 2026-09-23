"""Domain and application errors (spec section 33).

These carry no HTTP knowledge. Mapping to status codes / RFC 9457 problem
documents happens exclusively in ``interfaces.api.errors``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.enums import JobStatus, RunStatus
    from domain.value_objects.identifiers import JobId, ProjectId, RunId, WorkerId

__all__ = [
    "CandidateError",
    "ConcurrencyConflictError",
    "DomainError",
    "EntityNotFoundError",
    "IdempotencyConflictError",
    "InferenceError",
    "InvalidStateTransitionError",
    "JobLeaseExpiredError",
    "LLMTimeoutError",
    "NoCompatibleWorkerError",
    "OutputTruncatedError",
    "PlanValidationError",
    "ProjectAlreadyExistsError",
    "ProjectNotModifiableError",
    "ProjectUploadError",
    "RunCancelledError",
    "StructuredOutputError",
    "ToolExecutionError",
    "WorkerUnavailableError",
    "WorkspaceError",
]


class DomainError(Exception):
    """Base class for every error raised by the domain or application layers."""

    code: str = "domain_error"

    def __init__(self, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.details.items()))
        return f"{self.message} ({rendered})"


class EntityNotFoundError(DomainError):
    """A referenced aggregate does not exist."""

    code = "not_found"

    def __init__(self, entity: str, identifier: object) -> None:
        super().__init__(f"{entity} not found", entity=entity, id=str(identifier))
        self.entity = entity
        self.identifier = identifier


class InvalidStateTransitionError(DomainError):
    """An aggregate was asked to move between two incompatible states."""

    code = "invalid_state_transition"

    def __init__(self, entity: str, current: object, requested: object) -> None:
        super().__init__(
            f"{entity} cannot move from {current} to {requested}",
            entity=entity,
            current=str(current),
            requested=str(requested),
        )
        self.entity = entity
        self.current = current
        self.requested = requested


class WorkerUnavailableError(DomainError):
    """A specific worker is known but cannot take work right now."""

    code = "worker_unavailable"

    def __init__(self, worker_id: WorkerId, status: object | None = None) -> None:
        super().__init__(
            "worker is not available",
            worker_id=str(worker_id),
            status=str(status) if status is not None else None,
        )
        self.worker_id = worker_id


class NoCompatibleWorkerError(DomainError):
    """No registered worker satisfies the job's requirements.

    This is *retryable*: a worker may join at any moment, so the orchestrator
    backs off and re-schedules rather than failing the run.
    """

    code = "no_compatible_worker"

    def __init__(self, reason: str = "no compatible worker available", **details: Any) -> None:
        super().__init__(reason, **details)


class InferenceError(DomainError):
    """The inference engine answered with a failure, or could not be reached.

    Distinct from ``LLMTimeoutError`` (a deadline) and from
    ``StructuredOutputError`` (a well-formed answer that violates its schema):
    this is the engine itself misbehaving, which the retry policy treats as a
    reason to try a different worker.
    """

    code = "inference_failed"

    def __init__(
        self, message: str, *, status_code: int | None = None, model: str | None = None
    ) -> None:
        super().__init__(message, status_code=status_code, model=model)
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """Transient server-side conditions deserve another attempt elsewhere."""
        if self.status_code is None:
            return True
        return self.status_code in (408, 409, 425, 429) or self.status_code >= 500


class LLMTimeoutError(DomainError):
    """An inference request exceeded its deadline."""

    code = "llm_timeout"

    def __init__(self, timeout_seconds: float, *, model: str | None = None) -> None:
        super().__init__(
            f"inference timed out after {timeout_seconds:g}s",
            timeout_seconds=timeout_seconds,
            model=model,
        )
        self.timeout_seconds = timeout_seconds


class StructuredOutputError(DomainError):
    """The model produced output that does not satisfy the expected schema."""

    code = "structured_output_invalid"

    def __init__(
        self,
        message: str,
        *,
        schema: str,
        raw_output: str | None = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(message, schema=schema, attempt=attempt)
        self.schema = schema
        self.raw_output = raw_output
        self.attempt = attempt


class OutputTruncatedError(StructuredOutputError):
    """The answer ran out of room before it was finished.

    A subclass because it *is* a structured-output failure to everything that
    only wants to catch one thing, and its own type because the one useful
    reaction differs: a schema violation is worth re-asking with the violation
    attached, while a truncation re-asked under the same budget truncates at
    exactly the same place.
    """

    code = "output_truncated"


class PlanValidationError(DomainError):
    """A structurally valid plan that is semantically unusable (cycles, dangling deps)."""

    code = "plan_invalid"


class ToolExecutionError(DomainError):
    """A deterministic tool could not be executed.

    A tool that ran and reported a non-zero exit code is *not* an error: that is
    a normal, meaningful ``ToolResult``. This exception means the tool could not
    run at all (missing binary, sandbox refused, output limit breached).
    """

    code = "tool_execution_failed"

    def __init__(self, tool: str, message: str, **details: Any) -> None:
        super().__init__(message, tool=tool, **details)
        self.tool = tool


class WorkspaceError(DomainError):
    """An isolated workspace could not be created, mutated or released."""

    code = "workspace_error"


class JobLeaseExpiredError(DomainError):
    """A worker reported on a job whose lease it no longer holds.

    The result must be discarded: the job has already been reassigned.
    """

    code = "job_lease_expired"

    def __init__(self, job_id: JobId, *, holder: WorkerId | None = None) -> None:
        super().__init__(
            "job lease has expired or is held by another worker",
            job_id=str(job_id),
            holder=str(holder) if holder is not None else None,
        )
        self.job_id = job_id


class RunCancelledError(DomainError):
    """Work was abandoned because the run is cancelling or cancelled."""

    code = "run_cancelled"

    def __init__(self, run_id: RunId) -> None:
        super().__init__("run has been cancelled", run_id=str(run_id))
        self.run_id = run_id


class CandidateError(DomainError):
    """A candidate reached an unusable state."""

    code = "candidate_error"


class ConcurrencyConflictError(DomainError):
    """Two writers raced for the same aggregate and one lost.

    Exists so infrastructure can translate an optimistic-locking failure
    without leaking its ORM upwards. It is retryable: the loser re-reads and
    re-applies its decision.
    """

    code = "concurrency_conflict"

    def __init__(self, entity: str, identifier: object) -> None:
        super().__init__(f"{entity} was modified concurrently", entity=entity, id=str(identifier))
        self.entity = entity
        self.identifier = identifier


class IdempotencyConflictError(DomainError):
    """An idempotency key was reused with a different payload."""

    code = "idempotency_conflict"

    def __init__(self, key: str) -> None:
        super().__init__(
            "idempotency key was already used with a different request body",
            idempotency_key=key,
        )
        self.key = key


class JobNotRetryableError(DomainError):
    """A job exhausted its attempts or failed in a non-retryable way."""

    code = "job_not_retryable"

    def __init__(self, job_id: JobId, status: JobStatus, attempt: int) -> None:
        super().__init__(
            "job cannot be retried",
            job_id=str(job_id),
            status=str(status),
            attempt=attempt,
        )


class ProjectUploadError(DomainError):
    """An upload that cannot become a project: empty, malformed, or unsafe.

    Unsafe means a path that would land outside the project directory — a
    `../` in a zip entry, an absolute filename. Those are refused rather than
    sanitised: rewriting `../etc/passwd` to `etc/passwd` would create a project
    the caller did not upload.
    """

    code = "project_upload_invalid"


class ProjectAlreadyExistsError(DomainError):
    """A second upload under a name that already names a project.

    Creating a project by *reference* is idempotent on its name, because the
    same reference means the same code. An upload is not: two uploads under one
    name may carry different files, and returning the old project as if it were
    the new one would make the caller run their agents on code they did not
    send. Pick another name, or delete the old project.
    """

    code = "project_exists"

    def __init__(self, name: str, project_id: ProjectId) -> None:
        super().__init__(
            f"a project named {name!r} already exists", name=name, project_id=str(project_id)
        )
        self.name = name
        self.project_id = project_id


class ProjectNotModifiableError(DomainError):
    """A project was asked to change while its runs are still using it.

    A run reads the toolchain every time it validates a candidate, so replacing
    the commands under a live run would let one run judge its own candidates by
    two different definitions of "passing" — and the losing candidates would
    have been discarded for failing a check the winners never faced. The
    refusal names the runs to wait for (or cancel), because the caller cannot
    act on "later".
    """

    code = "project_not_modifiable"

    def __init__(self, project_id: ProjectId, active_run_ids: Sequence[RunId]) -> None:
        super().__init__(
            "project cannot be modified while runs are in flight",
            project_id=str(project_id),
            active_run_ids=[str(run_id) for run_id in active_run_ids],
        )
        self.project_id = project_id
        self.active_run_ids = tuple(active_run_ids)


class RunNotModifiableError(DomainError):
    """A mutation was attempted on a run in a terminal state."""

    code = "run_not_modifiable"

    def __init__(self, run_id: RunId, status: RunStatus) -> None:
        super().__init__(
            "run is in a terminal state",
            run_id=str(run_id),
            status=str(status),
        )
