"""Shared fixtures.

Time and identity are injected everywhere, so tests control both. Nothing here
touches a network, a database or a GPU.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from domain.entities.project import Project, ToolchainConfig
from domain.entities.worker import Worker
from domain.enums import AgentRole
from domain.value_objects.identifiers import ProjectId, WorkerId
from domain.value_objects.worker import GpuSpec, WorkerCapabilities, WorkerEndpoint
from infrastructure.database.engine import to_async_url


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        return self._now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def now(clock: FakeClock) -> datetime:
    return clock.now()


@pytest.fixture
def project(now: datetime) -> Project:
    return Project.create(
        project_id=ProjectId.generate(),
        name="demo",
        now=now,
        repository_url="https://example.invalid/demo.git",
        toolchain=ToolchainConfig(
            language="python", build_command="python -m compileall .", test_command="pytest -q"
        ),
    )


def make_worker(
    *,
    now: datetime,
    model_id: str = "Qwen3-Coder-30B-A3B",
    concurrency: int = 2,
    context_length: int = 262_144,
    roles: frozenset[AgentRole] = frozenset(AgentRole),
    endpoint: str = "http://worker:8000",
) -> Worker:
    return Worker.register(
        worker_id=WorkerId.generate(),
        endpoint=WorkerEndpoint(endpoint),
        capabilities=WorkerCapabilities(
            model_id=model_id,
            context_length=context_length,
            max_concurrency=concurrency,
            supported_roles=roles,
            gpu=GpuSpec(gpu_type="H200", gpu_count=1),
        ),
        now=now,
    )


@pytest.fixture
def worker(now: datetime) -> Worker:
    return make_worker(now=now)


# Pinned rather than ``latest``: the adapter relies on partial unique indexes
# and advisory locks, which must be exercised against the deployed version.
POSTGRES_IMAGE = os.environ.get("TEST_POSTGRES_IMAGE", "postgres:16-alpine")


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A usable PostgreSQL, from the environment or from a throwaway container.

    Defined at the root so both the adapter tests and the end-to-end scenario
    share one container instead of starting two.
    """
    provided = os.environ.get("TEST_DATABASE_URL")
    if provided:
        yield to_async_url(provided)
        return

    postgres = pytest.importorskip(
        "testcontainers.community.postgres", reason="testcontainers is not installed"
    )
    try:
        container = postgres.PostgresContainer(POSTGRES_IMAGE, driver="asyncpg")
        container.start()
    except Exception as exc:  # a missing daemon, a missing image, a refused socket
        pytest.skip(f"no PostgreSQL available: {exc}")
    try:
        yield str(container.get_connection_url())
    finally:
        container.stop()
