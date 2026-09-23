"""A whole control plane over HTTP, entirely in memory.

The API is exercised against the *real* use cases wired to the doubles from
``tests/application/fakes.py``. Stubbing the use cases instead would test the
routes against a fiction and would never catch a command built with the wrong
field — which is the only kind of bug this layer can have.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI
from tests.application.fakes import (
    FakeClock,
    FakeEventBus,
    FakeIdGenerator,
    FakeJobQueue,
    FakeWorkerRegistry,
    _Store,
    uow_factory_for,
)

from application.use_cases.approvals import ApproveRunUseCase
from application.use_cases.projects import (
    CreateProjectUseCase,
    GetProjectUseCase,
    ListProjectsUseCase,
    ReplaceProjectToolchainUseCase,
)
from application.use_cases.runs import (
    CancelRunUseCase,
    CreateRunUseCase,
    GetCandidatePatchUseCase,
    GetRunUseCase,
    ListCandidatesUseCase,
    ListReviewsUseCase,
    ListRunEventsUseCase,
    ListRunsUseCase,
)
from application.use_cases.workers import (
    DeregisterWorkerUseCase,
    DrainWorkerUseCase,
    HeartbeatUseCase,
    ListWorkersUseCase,
    RegisterWorkerUseCase,
)
from domain.entities.run import Run
from domain.exceptions import EntityNotFoundError
from domain.services.task_complexity import HeuristicTaskComplexityPolicy
from domain.value_objects.identifiers import RunId
from interfaces.api.app import create_api
from interfaces.api.dependencies.container import ApiDependencies
from interfaces.api.dependencies.readiness import DependencyHealth, ReadinessReport
from interfaces.worker_api.auth import SharedSecretServiceAuthenticator

SERVICE_TOKEN = "test-service-token"
HEARTBEAT_TTL = timedelta(seconds=30)


class ToggleReadinessProbe:
    """A readiness probe the test flips, like a dependency going down."""

    def __init__(self) -> None:
        self.healthy = True

    async def check(self) -> ReadinessReport:
        return ReadinessReport.of(
            [
                DependencyHealth(name="database", healthy=self.healthy),
                DependencyHealth(
                    name="redis",
                    healthy=self.healthy,
                    detail=None if self.healthy else "connection refused",
                ),
            ]
        )


@dataclass
class Harness:
    """Everything a test may need to arrange state or inspect the aftermath."""

    app: FastAPI
    store: _Store
    clock: FakeClock
    ids: FakeIdGenerator
    bus: FakeEventBus
    queue: FakeJobQueue
    registry: FakeWorkerRegistry
    probe: ToggleReadinessProbe
    headers: dict[str, str] = field(default_factory=dict)


class _NoApprovals:
    """Approval is an orchestrator operation; these are route tests.

    Refusing rather than returning a stub run: a test that reached this would
    be testing nothing, and should say so loudly.
    """

    async def approve(self, run_id: RunId) -> Run:
        raise EntityNotFoundError("Run", run_id)

    async def reject(self, run_id: RunId, *, reason: str) -> Run:
        raise EntityNotFoundError("Run", run_id)


@pytest.fixture
def harness() -> Harness:
    store = _Store()
    clock = FakeClock()
    ids = FakeIdGenerator()
    bus = FakeEventBus()
    queue = FakeJobQueue(clock)
    registry = FakeWorkerRegistry()
    probe = ToggleReadinessProbe()
    uow_factory = uow_factory_for(store)

    dependencies = ApiDependencies(
        create_project=CreateProjectUseCase(uow_factory=uow_factory, clock=clock, ids=ids),
        get_project=GetProjectUseCase(uow_factory=uow_factory),
        list_projects=ListProjectsUseCase(uow_factory=uow_factory),
        replace_project_toolchain=ReplaceProjectToolchainUseCase(uow_factory=uow_factory),
        create_run=CreateRunUseCase(
            uow_factory=uow_factory,
            bus=bus,
            clock=clock,
            ids=ids,
            complexity=HeuristicTaskComplexityPolicy(),
        ),
        get_run=GetRunUseCase(uow_factory=uow_factory),
        cancel_run=CancelRunUseCase(uow_factory=uow_factory, bus=bus, queue=queue, clock=clock),
        list_candidates=ListCandidatesUseCase(uow_factory=uow_factory),
        list_runs=ListRunsUseCase(uow_factory=uow_factory),
        candidate_patch=GetCandidatePatchUseCase(uow_factory=uow_factory),
        list_reviews=ListReviewsUseCase(uow_factory=uow_factory),
        approve_run=ApproveRunUseCase(orchestrator=_NoApprovals()),
        list_run_events=ListRunEventsUseCase(uow_factory=uow_factory),
        list_workers=ListWorkersUseCase(registry=registry),
        register_worker=RegisterWorkerUseCase(
            registry=registry, bus=bus, clock=clock, ids=ids, heartbeat_ttl=HEARTBEAT_TTL
        ),
        worker_heartbeat=HeartbeatUseCase(
            registry=registry, clock=clock, heartbeat_ttl=HEARTBEAT_TTL
        ),
        drain_worker=DrainWorkerUseCase(registry=registry, bus=bus, clock=clock),
        deregister_worker=DeregisterWorkerUseCase(registry=registry, bus=bus, clock=clock),
        event_bus=bus,
        readiness=probe,
        service_authenticator=SharedSecretServiceAuthenticator(SERVICE_TOKEN),
    )
    app = create_api(dependencies=dependencies)
    return Harness(
        app=app,
        store=store,
        clock=clock,
        ids=ids,
        bus=bus,
        queue=queue,
        registry=registry,
        probe=probe,
    )


@pytest.fixture
async def client(harness: Harness) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client speaking to the app in-process, without a socket."""
    transport = httpx.ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        yield client


@pytest.fixture
def service_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SERVICE_TOKEN}"}


async def create_project(client: httpx.AsyncClient, *, name: str = "demo") -> str:
    """Arrange a project and return its id; used by most run tests."""
    response = await client.post(
        "/v1/projects",
        json={"name": name, "repository_url": "https://example.invalid/demo.git"},
    )
    assert response.status_code == 201, response.text
    project_id: str = response.json()["id"]
    return project_id


async def create_run(
    client: httpx.AsyncClient,
    project_id: str,
    *,
    objective: str = "Implement the packet parser and add tests",
    idempotency_key: str | None = None,
    candidate_count: int | None = None,
) -> httpx.Response:
    payload: dict[str, object] = {"objective": objective}
    if candidate_count is not None:
        payload["candidate_count"] = candidate_count
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return await client.post(f"/v1/projects/{project_id}/runs", json=payload, headers=headers)
