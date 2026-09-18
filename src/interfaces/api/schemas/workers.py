"""Worker schemas: registration, heartbeat and inspection (spec sections 19, 46).

Registration payloads come from a worker process, which is authenticated but
still external: every declared capability is bounded and typed here before it
can influence a scheduling decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from application.dto.commands import (
    DeregisterWorkerCommand,
    DrainWorkerCommand,
    HeartbeatCommand,
    RegisterWorkerCommand,
)
from application.dto.views import WorkerView
from domain.enums import AgentRole, WorkerStatus
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import GpuSpec, WorkerLoad

__all__ = [
    "DrainWorkerRequest",
    "GpuPayload",
    "HeartbeatRequest",
    "RegisterWorkerRequest",
    "WorkerHealthResponse",
    "WorkerListResponse",
    "WorkerResponse",
    "deregister_command",
]

_ENDPOINT_PATTERN = r"^https?://\S+$"
"""The domain's ``WorkerEndpoint`` refuses anything else; matching it here keeps
a typo in a worker's configuration a 422 instead of a 500.
"""

MAX_CONTEXT_LENGTH = 10_000_000
MAX_CONCURRENCY = 1024


class GpuPayload(BaseModel):
    """Hardware the worker reports. Advisory: used for placement, never trusted."""

    model_config = ConfigDict(extra="forbid")

    gpu_type: str | None = Field(default=None, max_length=100)
    gpu_count: int = Field(default=1, ge=0, le=64)
    memory_gb: float | None = Field(default=None, ge=0)
    tensor_parallel_size: int = Field(default=1, ge=1, le=64)

    def to_spec(self) -> GpuSpec:
        return GpuSpec(
            gpu_type=self.gpu_type,
            gpu_count=self.gpu_count,
            memory_gb=self.memory_gb,
            tensor_parallel_size=self.tensor_parallel_size,
        )


class RegisterWorkerRequest(BaseModel):
    """A worker announcing itself to the control plane.

    ``worker_id`` is optional and supplied by the worker when it has a stable
    identity (a pod name, a RunPod instance id). Sending the same id again is a
    refresh, not a second worker — which is what makes a restart or a retried,
    timed-out call safe.
    """

    # ``model_id`` is the ubiquitous-language name for the served LLM; pydantic
    # reserves the ``model_`` prefix, so the namespace guard is lifted rather
    # than renaming a field that is part of the published contract.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    endpoint: str = Field(pattern=_ENDPOINT_PATTERN, max_length=2000)
    model_id: str = Field(min_length=1, max_length=200)
    context_length: int = Field(gt=0, le=MAX_CONTEXT_LENGTH)
    max_concurrency: int = Field(default=1, ge=1, le=MAX_CONCURRENCY)
    worker_id: UUID | None = None
    supported_roles: list[AgentRole] = Field(default_factory=lambda: list(AgentRole), min_length=1)
    gpu: GpuPayload = Field(default_factory=GpuPayload)
    supports_tools: bool = True
    supports_json_schema: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_command(self) -> RegisterWorkerCommand:
        return RegisterWorkerCommand(
            endpoint=self.endpoint,
            model_id=self.model_id,
            context_length=self.context_length,
            max_concurrency=self.max_concurrency,
            worker_id=WorkerId(self.worker_id) if self.worker_id else None,
            supported_roles=frozenset(self.supported_roles),
            gpu=self.gpu.to_spec(),
            supports_tools=self.supports_tools,
            supports_json_schema=self.supports_json_schema,
            metadata=dict(self.metadata),
        )


class HeartbeatRequest(BaseModel):
    """Liveness plus self-reported load.

    The worker reports its own occupancy because it is the only party that knows
    it: the control plane's view is always one scheduling decision behind.
    """

    model_config = ConfigDict(extra="forbid")

    active_jobs: int = Field(default=0, ge=0, le=MAX_CONCURRENCY)
    queued_jobs: int = Field(default=0, ge=0, le=MAX_CONCURRENCY)
    draining: bool = Field(
        default=False,
        description="Set once the worker has started shutting down and takes no new job.",
    )

    def to_command(self, *, worker_id: UUID) -> HeartbeatCommand:
        return HeartbeatCommand(
            worker_id=WorkerId(worker_id),
            load=WorkerLoad(active_jobs=self.active_jobs, queued_jobs=self.queued_jobs),
            draining=self.draining,
        )


class DrainWorkerRequest(BaseModel):
    """Drain has no parameters today; the model exists so adding one is not a
    breaking change for clients already sending ``{}``."""

    model_config = ConfigDict(extra="forbid")

    def to_command(self, *, worker_id: UUID) -> DrainWorkerCommand:
        return DrainWorkerCommand(worker_id=WorkerId(worker_id))


def deregister_command(*, worker_id: UUID, graceful: bool) -> DeregisterWorkerCommand:
    """``DELETE`` carries no body, so its single option is a query parameter."""
    return DeregisterWorkerCommand(worker_id=WorkerId(worker_id), graceful=graceful)


class WorkerResponse(BaseModel):
    """A worker as the control plane knows it."""

    model_config = ConfigDict(protected_namespaces=())

    id: UUID
    model_id: str
    status: WorkerStatus
    endpoint: str
    capacity: int
    active_jobs: int
    context_length: int
    gpu_type: str | None
    gpu_count: int
    supported_roles: list[str]
    registered_at: datetime
    last_heartbeat_at: datetime

    @classmethod
    def of(cls, view: WorkerView) -> WorkerResponse:
        return cls(
            id=view.id.value,
            model_id=view.model_id,
            status=view.status,
            endpoint=view.endpoint,
            capacity=view.capacity,
            active_jobs=view.active_jobs,
            context_length=view.context_length,
            gpu_type=view.gpu_type,
            gpu_count=view.gpu_count,
            supported_roles=list(view.supported_roles),
            registered_at=view.registered_at,
            last_heartbeat_at=view.last_heartbeat_at,
        )


class WorkerListResponse(BaseModel):
    workers: list[WorkerResponse]

    @classmethod
    def of(cls, views: Sequence[WorkerView]) -> WorkerListResponse:
        return cls(workers=[WorkerResponse.of(view) for view in views])


class WorkerHealthResponse(BaseModel):
    """Health of one worker, for an operator or the worker's own supervisor.

    ``live`` is read from ``WorkerStatus.is_live``: the domain decides what
    "live" means, the API only reports it.
    """

    id: UUID
    status: WorkerStatus
    live: bool
    accepts_new_jobs: bool
    active_jobs: int
    capacity: int
    last_heartbeat_at: datetime

    @classmethod
    def of(cls, view: WorkerView) -> WorkerHealthResponse:
        return cls(
            id=view.id.value,
            status=view.status,
            live=view.status.is_live,
            accepts_new_jobs=view.status.accepts_new_jobs,
            active_jobs=view.active_jobs,
            capacity=view.capacity,
            last_heartbeat_at=view.last_heartbeat_at,
        )
