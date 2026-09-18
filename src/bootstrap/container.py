"""The composition root.

The single place that knows every concrete implementation at once. Nothing else
imports an adapter: they all receive their collaborators, which is what makes
the inner layers testable and the outer ones replaceable.

Read this file to learn what the platform is actually made of today.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from application.orchestration.agents import (
    CoderAgent,
    PlannerAgent,
    ReviewerAgent,
    StructuredCompletion,
)
from application.orchestration.executor import ExecutorConfig, JobExecutor
from application.orchestration.maintenance import MaintenanceConfig, MaintenanceLoop
from application.orchestration.orchestrator import OrchestratorConfig, RunOrchestrator
from application.orchestration.worker_pool import WorkerPool
from application.ports import UnitOfWorkFactory
from application.services.locking import InProcessRunCoordinator
from application.use_cases.projects import (
    CreateProjectUseCase,
    GetProjectUseCase,
    ListProjectsUseCase,
)
from application.use_cases.runs import (
    CancelRunUseCase,
    CreateRunUseCase,
    GetRunUseCase,
    ListCandidatesUseCase,
    ListRunEventsUseCase,
    ListRunsUseCase,
)
from application.use_cases.workers import (
    DeregisterWorkerUseCase,
    DrainWorkerUseCase,
    HeartbeatUseCase,
    ListWorkersUseCase,
    ReapStaleWorkersUseCase,
    RegisterWorkerUseCase,
)
from bootstrap.adapters import PromptLibraryRenderer, StructuredOutputCodec
from bootstrap.config import LLMProviderKind, SchedulerStrategy, Settings
from bootstrap.identity import SystemClock, UuidGenerator
from bootstrap.tooling import ProjectToolExecutorFactory
from domain.ports.event_bus import EventBus
from domain.ports.job_queue import JobQueue
from domain.ports.llm_provider import LLMProviderFactory
from domain.ports.worker_registry import WorkerRegistry
from domain.ports.worker_scheduler import WorkerScheduler
from domain.ports.workspace import WorkspaceManager
from domain.services.scheduling import LeastLoadedCompatibleScheduler, RoundRobinScheduler
from domain.services.task_complexity import HeuristicTaskComplexityPolicy
from domain.value_objects.identifiers import WorkerId
from infrastructure.database import (
    SqlAlchemyUnitOfWork,
    create_database_engine,
    create_session_factory,
)
from infrastructure.llm import (
    FakeLLMProviderFactory,
    HttpLLMProviderFactory,
    OpenAICompatibleSettings,
    PromptLibrary,
)
from infrastructure.queue import (
    InMemoryEventBus,
    InMemoryJobQueue,
    InMemoryWorkerRegistry,
    RedisJobQueue,
)
from infrastructure.redis import RedisEventBus, RedisWorkerRegistry, create_redis_client
from infrastructure.tools import RipgrepRepositoryContextProvider, create_sandbox_executor
from infrastructure.workspace import GitWorktreeWorkspaceManager

__all__ = ["Container", "build_container"]

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    """Everything the processes need, already wired."""

    settings: Settings
    clock: SystemClock
    ids: UuidGenerator
    uow_factory: UnitOfWorkFactory
    registry: WorkerRegistry
    queue: JobQueue
    bus: EventBus
    scheduler: WorkerScheduler
    workspaces: WorkspaceManager
    llm_factory: LLMProviderFactory
    orchestrator: RunOrchestrator
    executor: JobExecutor
    maintenance: MaintenanceLoop

    # use cases
    create_project: CreateProjectUseCase
    get_project: GetProjectUseCase
    list_projects: ListProjectsUseCase
    create_run: CreateRunUseCase
    cancel_run: CancelRunUseCase
    get_run: GetRunUseCase
    list_runs: ListRunsUseCase
    list_candidates: ListCandidatesUseCase
    list_run_events: ListRunEventsUseCase
    register_worker: RegisterWorkerUseCase
    heartbeat: HeartbeatUseCase
    drain_worker: DrainWorkerUseCase
    deregister_worker: DeregisterWorkerUseCase
    list_workers: ListWorkersUseCase
    reap_workers: ReapStaleWorkersUseCase

    _closers: list[Any] = field(default_factory=list, repr=False)

    async def aclose(self) -> None:
        """Release every resource, tolerating partial construction."""
        for closer in reversed(self._closers):
            try:
                await closer()
            except Exception:  # shutdown must not mask the original failure
                _log.warning("failed to close a resource cleanly", exc_info=True)
        self._closers.clear()


def _build_scheduler(strategy: SchedulerStrategy) -> WorkerScheduler:
    if strategy is SchedulerStrategy.ROUND_ROBIN:
        return RoundRobinScheduler()
    return LeastLoadedCompatibleScheduler()


def _build_llm_factory(settings: Settings) -> LLMProviderFactory:
    """Pick the inference adapter.

    The fake is not a testing shortcut: it is how the whole stack runs locally
    and in CI without a GPU. Configuration refuses it in production.
    """
    if settings.llm_provider is LLMProviderKind.FAKE:
        return FakeLLMProviderFactory(
            model_id=settings.model_id, context_length=settings.model_context_length
        )
    return HttpLLMProviderFactory(
        OpenAICompatibleSettings(
            context_length=settings.model_context_length,
            default_timeout_seconds=settings.llm_request_timeout_seconds,
        )
    )


async def build_container(
    settings: Settings,
    *,
    in_memory_messaging: bool = False,
    workspaces: WorkspaceManager | None = None,
    llm_factory: LLMProviderFactory | None = None,
) -> Container:
    """Assemble the platform.

    ``in_memory_messaging`` swaps Redis for equivalent in-process adapters. They
    honour the same ports with the same semantics — leases, priorities, expiry —
    which is what lets an end-to-end test exercise the real orchestrator without
    a broker.
    """
    clock = SystemClock()
    ids = UuidGenerator()
    closers: list[Any] = []

    engine = create_database_engine(settings.database_url)
    closers.append(engine.dispose)
    session_factory = create_session_factory(engine)

    def uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    if in_memory_messaging:
        registry: WorkerRegistry = InMemoryWorkerRegistry()
        queue: JobQueue = InMemoryJobQueue()
        bus: EventBus = InMemoryEventBus()
    else:
        redis = create_redis_client(settings.redis_url)
        closers.append(redis.close)
        registry = RedisWorkerRegistry(redis)
        queue = RedisJobQueue(redis)
        bus = RedisEventBus(redis)

    sandbox = create_sandbox_executor(prefer_docker=not settings.environment.is_local)
    tools = ProjectToolExecutorFactory(sandbox=sandbox)
    context = RipgrepRepositoryContextProvider(sandbox=sandbox)
    workspace_manager = workspaces or GitWorktreeWorkspaceManager(root=settings.workspace_root)

    codec = StructuredOutputCodec()
    renderer = PromptLibraryRenderer(library=PromptLibrary(settings.prompts_root), codec=codec)
    completion = StructuredCompletion(
        renderer=renderer,
        codec=codec,
        timeout_seconds=settings.llm_request_timeout_seconds,
    )

    provider_factory = llm_factory or _build_llm_factory(settings)
    scheduler = _build_scheduler(settings.scheduler_strategy)
    pool = WorkerPool(registry=registry, scheduler=scheduler, factory=provider_factory)

    orchestrator = RunOrchestrator(
        uow_factory=uow_factory,
        bus=bus,
        queue=queue,
        pool=pool,
        coordinator=InProcessRunCoordinator(),
        workspaces=workspace_manager,
        tools=tools,
        context=context,
        planner=PlannerAgent(completion=completion),
        coder=CoderAgent(completion=completion),
        reviewer=ReviewerAgent(completion=completion),
        clock=clock,
        ids=ids,
        config=OrchestratorConfig(
            lease_duration=settings.job_lease,
            job_max_attempts=settings.job_max_attempts,
            static_analysis_is_blocking=settings.static_analysis_is_blocking,
        ),
    )
    executor = JobExecutor(
        queue=queue,
        orchestrator=orchestrator,
        clock=clock,
        executor_id=ids.next_id(WorkerId),
        config=ExecutorConfig(
            concurrency=settings.executor_concurrency, lease_duration=settings.job_lease
        ),
    )
    reaper = ReapStaleWorkersUseCase(
        registry=registry, clock=clock, heartbeat_timeout=settings.heartbeat_timeout
    )
    maintenance = MaintenanceLoop(
        queue=queue,
        uow_factory=uow_factory,
        bus=bus,
        clock=clock,
        reaper=reaper,
        orchestrator=orchestrator,
        config=MaintenanceConfig(interval=settings.reaper_interval),
    )

    return Container(
        settings=settings,
        clock=clock,
        ids=ids,
        uow_factory=uow_factory,
        registry=registry,
        queue=queue,
        bus=bus,
        scheduler=scheduler,
        workspaces=workspace_manager,
        llm_factory=provider_factory,
        orchestrator=orchestrator,
        executor=executor,
        maintenance=maintenance,
        create_project=CreateProjectUseCase(uow_factory=uow_factory, clock=clock, ids=ids),
        get_project=GetProjectUseCase(uow_factory=uow_factory),
        list_projects=ListProjectsUseCase(uow_factory=uow_factory),
        create_run=CreateRunUseCase(
            uow_factory=uow_factory,
            bus=bus,
            clock=clock,
            ids=ids,
            complexity=HeuristicTaskComplexityPolicy(),
            limits=settings.run_limits,
        ),
        cancel_run=CancelRunUseCase(uow_factory=uow_factory, bus=bus, queue=queue, clock=clock),
        get_run=GetRunUseCase(uow_factory=uow_factory),
        list_runs=ListRunsUseCase(uow_factory=uow_factory),
        list_candidates=ListCandidatesUseCase(uow_factory=uow_factory),
        list_run_events=ListRunEventsUseCase(uow_factory=uow_factory),
        register_worker=RegisterWorkerUseCase(
            registry=registry,
            bus=bus,
            clock=clock,
            ids=ids,
            heartbeat_ttl=settings.heartbeat_ttl,
        ),
        heartbeat=HeartbeatUseCase(
            registry=registry, clock=clock, heartbeat_ttl=settings.heartbeat_ttl
        ),
        drain_worker=DrainWorkerUseCase(registry=registry, bus=bus, clock=clock),
        deregister_worker=DeregisterWorkerUseCase(registry=registry, bus=bus, clock=clock),
        list_workers=ListWorkersUseCase(registry=registry),
        reap_workers=reaper,
        _closers=closers,
    )


def describe(container: Container) -> Sequence[str]:
    """One line per wired adapter, logged at startup.

    Knowing which implementations are live is the first question asked when a
    deployment misbehaves.
    """
    return (
        f"database={container.settings.database_url.split('@')[-1]}",
        f"registry={type(container.registry).__name__}",
        f"queue={type(container.queue).__name__}",
        f"event_bus={type(container.bus).__name__}",
        f"scheduler={container.scheduler.name}",
        f"workspaces={type(container.workspaces).__name__}",
        f"inference={type(container.llm_factory).__name__}",
        f"model={container.settings.model_id}",
    )
