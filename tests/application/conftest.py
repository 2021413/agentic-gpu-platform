"""Wiring for the application tests: a whole platform, entirely in memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pytest
from tests.application.fakes import (
    FakeClock,
    FakeContextProvider,
    FakeEventBus,
    FakeIdGenerator,
    FakeJobQueue,
    FakeLLMProvider,
    FakeLLMProviderFactory,
    FakeOutputCodec,
    FakePromptRenderer,
    FakeToolExecutor,
    FakeWorkerRegistry,
    FakeWorkspaceManager,
    _Store,
    uow_factory_for,
)

from application.orchestration.agents import (
    CoderAgent,
    PlannerAgent,
    ReviewerAgent,
    StructuredCompletion,
)
from application.orchestration.executor import ExecutorConfig, JobExecutor
from application.orchestration.orchestrator import OrchestratorConfig, RunOrchestrator
from application.orchestration.worker_pool import WorkerPool
from application.services.locking import InProcessRunCoordinator
from domain.entities.project import Project, ToolchainConfig
from domain.entities.worker import Worker
from domain.enums import AgentRole
from domain.services.scheduling import LeastLoadedCompatibleScheduler
from domain.value_objects.identifiers import ProjectId, WorkerId
from domain.value_objects.worker import GpuSpec, WorkerCapabilities, WorkerEndpoint


@dataclass
class Platform:
    """Every collaborator a test may want to inspect or tamper with."""

    store: _Store
    clock: FakeClock
    ids: FakeIdGenerator
    bus: FakeEventBus
    queue: FakeJobQueue
    registry: FakeWorkerRegistry
    workspaces: FakeWorkspaceManager
    tools: FakeToolExecutor
    provider: FakeLLMProvider
    orchestrator: RunOrchestrator
    executor: JobExecutor
    uow_factory: object

    async def drain(self, *, max_steps: int = 40) -> int:
        """Execute queued jobs until the run settles.

        The bound is a safety net: an orchestrator that keeps scheduling work
        forever is a bug, and the test must fail rather than hang.
        """
        steps = 0
        while await self.executor.run_once():
            steps += 1
            if steps >= max_steps:
                raise AssertionError(
                    f"the workflow did not settle within {max_steps} jobs; "
                    "the orchestrator is looping"
                )
        return steps

    async def add_worker(
        self,
        *,
        concurrency: int = 2,
        roles: frozenset[AgentRole] = frozenset(AgentRole),
        model_id: str = "fake-coder",
    ) -> Worker:
        worker = Worker.register(
            worker_id=self.ids.next_id(WorkerId),
            endpoint=WorkerEndpoint(f"http://worker-{len(self.registry.workers)}:8000"),
            capabilities=WorkerCapabilities(
                model_id=model_id,
                context_length=32_000,
                max_concurrency=concurrency,
                supported_roles=roles,
                gpu=GpuSpec(gpu_type="H200", gpu_count=1),
            ),
            now=self.clock.now(),
        )
        await self.registry.register(worker, ttl=timedelta(seconds=30))
        return worker


@pytest.fixture
def platform(request: pytest.FixtureRequest) -> Platform:
    marker = request.node.get_closest_marker("tool_exit_codes")
    exit_codes = marker.args[0] if marker else {}

    clock = FakeClock()
    ids = FakeIdGenerator()
    store = _Store()
    bus = FakeEventBus()
    queue = FakeJobQueue(clock)
    registry = FakeWorkerRegistry()
    workspaces = FakeWorkspaceManager(ids)
    tools = FakeToolExecutor(exit_codes)
    provider = FakeLLMProvider()
    factory = FakeLLMProviderFactory(provider)

    completion = StructuredCompletion(renderer=FakePromptRenderer(), codec=FakeOutputCodec())
    orchestrator = RunOrchestrator(
        uow_factory=uow_factory_for(store),
        bus=bus,
        queue=queue,
        pool=WorkerPool(
            registry=registry, scheduler=LeastLoadedCompatibleScheduler(), factory=factory
        ),
        coordinator=InProcessRunCoordinator(),
        workspaces=workspaces,
        tools=tools,
        context=FakeContextProvider(),
        planner=PlannerAgent(completion=completion),
        coder=CoderAgent(completion=completion),
        reviewer=ReviewerAgent(completion=completion),
        clock=clock,
        ids=ids,
        config=OrchestratorConfig(),
    )
    executor = JobExecutor(
        queue=queue,
        orchestrator=orchestrator,
        clock=clock,
        executor_id=WorkerId.generate(),
        config=ExecutorConfig(concurrency=4),
    )
    return Platform(
        store=store,
        clock=clock,
        ids=ids,
        bus=bus,
        queue=queue,
        registry=registry,
        workspaces=workspaces,
        tools=tools,
        provider=provider,
        orchestrator=orchestrator,
        executor=executor,
        uow_factory=uow_factory_for(store),
    )


@pytest.fixture
async def project(platform: Platform) -> Project:
    project = Project.create(
        project_id=platform.ids.next_id(ProjectId),
        name="demo",
        now=platform.clock.now(),
        repository_url="https://example.invalid/demo.git",
        toolchain=ToolchainConfig(
            language="python",
            build_command="python -m compileall .",
            test_command="pytest -q",
        ),
    )
    await platform.store.projects.add(project)
    return project
