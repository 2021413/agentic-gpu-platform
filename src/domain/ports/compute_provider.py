"""Optional GPU provisioning (spec section 24).

RunPod and friends stay behind this port. The base platform never provisions
anything: workers connect to the control plane on their own. Defining the port
now is what keeps autoscaling an additive change later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from domain.enums import WorkerStatus
from domain.value_objects.identifiers import WorkerId

__all__ = ["ComputeProvider", "WorkerHandle", "WorkerSpec"]


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """What to provision. Provider-neutral on purpose."""

    model_id: str
    gpu_type: str | None = None
    gpu_count: int = 1
    image: str | None = None
    region: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    max_hourly_cost: float | None = None


@dataclass(frozen=True, slots=True)
class WorkerHandle:
    """A provisioned instance, before or after it registers itself."""

    provider: str
    external_id: str
    worker_id: WorkerId | None = None
    status: WorkerStatus = WorkerStatus.STARTING
    endpoint: str | None = None
    hourly_cost: float | None = None


@runtime_checkable
class ComputeProvider(Protocol):
    """Provisions and terminates GPU instances."""

    @property
    def name(self) -> str: ...

    async def provision_worker(self, spec: WorkerSpec) -> WorkerHandle: ...
    async def terminate_worker(self, worker_id: WorkerId) -> None: ...
    async def list_workers(self) -> Sequence[WorkerHandle]: ...
