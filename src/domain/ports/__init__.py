"""Ports: the interfaces through which the domain reaches the outside world.

Every one of these is implemented in ``infrastructure`` and injected at
composition time. The dependency arrow always points inwards: nothing in this
package knows that PostgreSQL, Redis, FastAPI or vLLM exist.
"""

from __future__ import annotations

from domain.ports.artifact_store import ArtifactRef, ArtifactStore
from domain.ports.clock import Clock, IdGenerator
from domain.ports.compute_provider import ComputeProvider, WorkerHandle, WorkerSpec
from domain.ports.event_bus import EventBus
from domain.ports.job_queue import JobQueue
from domain.ports.llm_provider import LLMProvider, LLMProviderFactory
from domain.ports.repositories import (
    CandidateRepository,
    EventStore,
    JobRepository,
    PlanRepository,
    ProjectRepository,
    ReviewRepository,
    RunRepository,
    ToolResultRepository,
    UnitOfWork,
)
from domain.ports.repository_context import (
    ContextRequest,
    FileExcerpt,
    RepositoryContext,
    RepositoryContextProvider,
)
from domain.ports.tools import SandboxExecutor, Tool, ToolExecutor, ToolRegistry
from domain.ports.worker_registry import WorkerRegistry
from domain.ports.worker_scheduler import WorkerScheduler
from domain.ports.workspace import WorkspaceManager

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "CandidateRepository",
    "Clock",
    "ComputeProvider",
    "ContextRequest",
    "EventBus",
    "EventStore",
    "FileExcerpt",
    "IdGenerator",
    "JobQueue",
    "JobRepository",
    "LLMProvider",
    "LLMProviderFactory",
    "PlanRepository",
    "ProjectRepository",
    "RepositoryContext",
    "RepositoryContextProvider",
    "ReviewRepository",
    "RunRepository",
    "SandboxExecutor",
    "Tool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResultRepository",
    "UnitOfWork",
    "WorkerHandle",
    "WorkerRegistry",
    "WorkerScheduler",
    "WorkerSpec",
    "WorkspaceManager",
]
