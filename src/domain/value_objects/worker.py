"""Value objects describing a GPU worker and what it can take on.

The orchestrator never asks "is this worker a vLLM server?"; it asks "does this
worker satisfy the requirements of that job?". Everything needed to answer lives
here, so eligibility is decidable without any network call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from domain.enums import AgentRole

__all__ = [
    "GpuSpec",
    "JobRequirements",
    "WorkerCapabilities",
    "WorkerEndpoint",
    "WorkerLoad",
]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    """Where the control plane reaches a worker. Opaque to the domain."""

    url: str

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"worker endpoint must be an http(s) URL, got {self.url!r}")

    def __str__(self) -> str:
        return self.url


@dataclass(frozen=True, slots=True)
class GpuSpec:
    """Hardware the worker reports at registration. Advisory, never trusted for safety."""

    gpu_type: str | None = None
    gpu_count: int = 1
    memory_gb: float | None = None
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError("gpu_count must not be negative")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")


@dataclass(frozen=True, slots=True)
class WorkerCapabilities:
    """What a worker declares it can do.

    ``max_concurrency`` is the worker's own statement of how many jobs it will
    accept at once; the scheduler treats it as a hard ceiling.
    """

    model_id: str
    context_length: int
    max_concurrency: int = 1
    supported_roles: frozenset[AgentRole] = frozenset(AgentRole)
    gpu: GpuSpec = field(default_factory=GpuSpec)
    supports_tools: bool = True
    supports_json_schema: bool = True
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if not self.supported_roles:
            raise ValueError("a worker must support at least one role")

    def supports_role(self, role: AgentRole) -> bool:
        return role in self.supported_roles

    def fits(self, estimated_tokens: int) -> bool:
        """Whether a prompt of that size plausibly fits this worker's context window."""
        return estimated_tokens <= self.context_length


@dataclass(frozen=True, slots=True)
class WorkerLoad:
    """Instantaneous occupancy, refreshed by heartbeats and by job assignment."""

    active_jobs: int = 0
    queued_jobs: int = 0

    def __post_init__(self) -> None:
        if self.active_jobs < 0 or self.queued_jobs < 0:
            raise ValueError("load counters must not be negative")

    @property
    def total(self) -> int:
        return self.active_jobs + self.queued_jobs


@dataclass(frozen=True, slots=True)
class JobRequirements:
    """What a job needs from a worker, expressed without naming any worker.

    ``model_id`` is optional on purpose: a run normally accepts whatever model
    the pool serves, and only pins one when reproducibility demands it.
    """

    role: AgentRole
    model_id: str | None = None
    estimated_prompt_tokens: int = 0
    requires_tools: bool = False
    requires_json_schema: bool = True

    def __post_init__(self) -> None:
        if self.estimated_prompt_tokens < 0:
            raise ValueError("estimated_prompt_tokens must not be negative")
