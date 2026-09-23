"""Commands: intents entering the application layer.

Every externally triggered mutation carries an optional idempotency key,
because delivery is assumed to be at-least-once (spec section 35).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from domain.entities.project import ToolchainConfig
from domain.enums import AgentRole, FailureKind
from domain.value_objects.identifiers import (
    IdempotencyKey,
    JobId,
    ProjectId,
    RunId,
    WorkerId,
)
from domain.value_objects.lease import LeaseToken
from domain.value_objects.worker import GpuSpec, WorkerLoad

__all__ = [
    "CancelRunCommand",
    "CreateProjectCommand",
    "CreateRunCommand",
    "DeregisterWorkerCommand",
    "DrainWorkerCommand",
    "HeartbeatCommand",
    "RegisterWorkerCommand",
    "ReplaceProjectToolchainCommand",
    "ReportJobFailureCommand",
    "ReportJobResultCommand",
    "UploadProjectCommand",
    "UploadedFile",
]


@dataclass(frozen=True, slots=True)
class CreateProjectCommand:
    name: str
    repository_url: str | None = None
    local_path: str | None = None
    default_branch: str = "main"
    toolchain: ToolchainConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """One file from an upload. ``path`` is relative to the project root."""

    path: str
    content: bytes


@dataclass(frozen=True, slots=True)
class UploadProjectCommand:
    """Create a project from the files the caller sent, and nothing else.

    There is no path here on purpose. Where the files end up is the server's
    decision, and the toolchain is detected from them unless the caller says
    otherwise for a specific field.
    """

    name: str
    files: Sequence[UploadedFile]
    language: str | None = None
    build_command: str | None = None
    test_command: str | None = None
    default_branch: str = "main"


@dataclass(frozen=True, slots=True)
class ReplaceProjectToolchainCommand:
    """Correct the commands a project runs.

    A whole ``ToolchainConfig``, never a handful of fields to merge: the
    commands are read together when a candidate is validated, and a partial
    update cannot tell "leave the test command alone" apart from "there is no
    test command" without inventing a sentinel for absence.
    """

    project_id: ProjectId
    toolchain: ToolchainConfig


@dataclass(frozen=True, slots=True)
class CreateRunCommand:
    """Start an agentic run.

    ``candidate_count`` is optional: left unset, the task-complexity policy
    decides how many candidates the objective is worth.
    """

    project_id: ProjectId
    objective: str
    candidate_count: int | None = None
    idempotency_key: IdempotencyKey | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CancelRunCommand:
    run_id: RunId
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RegisterWorkerCommand:
    """A worker announcing itself. Re-registration with the same id is a refresh."""

    endpoint: str
    model_id: str
    context_length: int
    max_concurrency: int = 1
    worker_id: WorkerId | None = None
    supported_roles: frozenset[AgentRole] = frozenset(AgentRole)
    gpu: GpuSpec = field(default_factory=GpuSpec)
    supports_tools: bool = True
    supports_json_schema: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HeartbeatCommand:
    worker_id: WorkerId
    load: WorkerLoad = field(default_factory=WorkerLoad)
    draining: bool = False


@dataclass(frozen=True, slots=True)
class DrainWorkerCommand:
    worker_id: WorkerId


@dataclass(frozen=True, slots=True)
class DeregisterWorkerCommand:
    worker_id: WorkerId
    graceful: bool = True


@dataclass(frozen=True, slots=True)
class ReportJobResultCommand:
    """A worker reporting success. The lease token proves it still owns the job."""

    job_id: JobId
    token: LeaseToken
    result: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReportJobFailureCommand:
    job_id: JobId
    token: LeaseToken
    kind: FailureKind
    reason: str
