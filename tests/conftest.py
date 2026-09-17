"""Shared fixtures.

Time and identity are injected everywhere, so tests control both. Nothing here
touches a network, a database or a GPU.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from domain.entities.project import Project, ToolchainConfig
from domain.entities.worker import Worker
from domain.enums import AgentRole
from domain.value_objects.identifiers import ProjectId, WorkerId
from domain.value_objects.worker import GpuSpec, WorkerCapabilities, WorkerEndpoint


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
