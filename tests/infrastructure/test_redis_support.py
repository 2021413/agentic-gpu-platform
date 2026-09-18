"""Shared plumbing for the infrastructure contract suites.

Not a test module despite the name: pytest only adds a directory to the import
path for files it collects, so the helpers the contract suites import live in a
file matching the collection pattern. It holds three things:

* the Redis endpoint — an already running server via ``REDIS_URL``, otherwise a
  throwaway container, otherwise a clean skip. An absent Docker daemon must
  make the integration variants disappear, never fail;
* one connection per adapter fixture, flushed before and after, so a test can
  never inherit another's keys;
* builders for the domain objects the suites need, kept here so both the
  in-memory and the Redis run are handed byte-identical inputs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from redis.asyncio import Redis

from domain.entities.job import Job
from domain.entities.worker import Worker
from domain.enums import AgentRole, JobType, Priority
from domain.value_objects.identifiers import JobId, ProjectId, RunId, WorkerId
from domain.value_objects.worker import (
    GpuSpec,
    JobRequirements,
    WorkerCapabilities,
    WorkerEndpoint,
)

try:  # pragma: no cover - the dev extra may be absent on a bare checkout
    from testcontainers.community.redis import RedisContainer
except ImportError:
    RedisContainer = None

REDIS_IMAGE = "redis:7-alpine"

MEMORY = "memory"
REDIS = "redis"

BACKENDS = [
    pytest.param(MEMORY, id="in-memory"),
    pytest.param(REDIS, id="redis", marks=pytest.mark.integration),
]
"""The two implementations every contract test runs against.

Marking only the Redis variant means ``-m 'not integration'`` runs the whole
suite offline, and a machine without Docker still proves the semantics.
"""

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
"""Fixed instant. Every operation in these ports takes ``now`` explicitly, so
the suite never sleeps to make time pass."""


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """A Redis to run the integration variants against, or a skip."""
    configured = os.environ.get("REDIS_URL")
    if configured:
        yield configured
        return
    if RedisContainer is None:  # pragma: no cover - depends on the dev extra
        pytest.skip("testcontainers is not installed")
    try:
        # Building the container already talks to the Docker daemon, so the
        # guard has to cover construction as well as startup.
        container = RedisContainer(REDIS_IMAGE)
        container.start()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no Redis available for integration tests: {type(exc).__name__}: {exc}")
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"
    finally:
        container.stop()


def connect(url: str) -> Redis:
    """A client with the settings every adapter here assumes.

    ``decode_responses`` is what the adapters are built for; the driver's own
    types cannot express it, which is why they narrow every reply themselves.
    """
    return Redis.from_url(url, decode_responses=True)


# -- domain builders ----------------------------------------------------
def make_job(
    *,
    run_id: RunId | None = None,
    job_type: JobType = JobType.CODE,
    priority: Priority = Priority.NORMAL,
    created_at: datetime = T0,
    max_attempts: int = 3,
    payload: dict[str, Any] | None = None,
    job_id: JobId | None = None,
    project_id: ProjectId | None = None,
) -> Job:
    """A queued job, ready to be handed to ``JobQueue.enqueue``."""
    job = Job.create(
        job_id=job_id or JobId.generate(),
        run_id=run_id or RunId.generate(),
        project_id=project_id or ProjectId.generate(),
        job_type=job_type,
        now=created_at,
        role=AgentRole.CODER if job_type.requires_inference else None,
        priority=priority,
        payload=payload or {"objective": "add a healthcheck"},
        requirements=JobRequirements(
            role=AgentRole.CODER, estimated_prompt_tokens=1024, requires_tools=True
        ),
        max_attempts=max_attempts,
    )
    job.enqueue(created_at)
    job.pull_events()
    return job


def make_worker(
    *,
    now: datetime = T0,
    concurrency: int = 2,
    model_id: str = "Qwen3-Coder-30B-A3B",
    roles: frozenset[AgentRole] = frozenset(AgentRole),
    worker_id: WorkerId | None = None,
) -> Worker:
    worker = Worker.register(
        worker_id=worker_id or WorkerId.generate(),
        endpoint=WorkerEndpoint("http://worker-1:8000"),
        capabilities=WorkerCapabilities(
            model_id=model_id,
            context_length=262_144,
            max_concurrency=concurrency,
            supported_roles=roles,
            gpu=GpuSpec(gpu_type="H200", gpu_count=1, memory_gb=141.0),
            metadata={"region": "eu-west"},
        ),
        now=now,
        metadata={"pod": "runpod-7"},
    )
    worker.pull_events()
    return worker


def later(seconds: float) -> datetime:
    """A moment after ``T0``; the suite's only way of moving time."""
    return T0 + timedelta(seconds=seconds)
