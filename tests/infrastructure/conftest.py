"""Fixtures for the PostgreSQL adapter tests.

Integration tests need a real PostgreSQL: the adapter relies on JSONB, partial
unique indexes and advisory locks, none of which SQLite can emulate, so testing
against a stand-in would prove nothing about what runs in production.

When no container runtime is available the fixtures skip instead of failing —
CI without Docker still runs the pure mapper tests, which need no server.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan, PlanTask
from domain.entities.project import Project
from domain.entities.review import Review, ReviewFinding, Severity
from domain.entities.run import Run
from domain.enums import AgentRole, JobType, Priority, ReviewVerdict
from domain.value_objects.identifiers import (
    CandidateId,
    IdempotencyKey,
    JobId,
    PlanId,
    ReviewId,
    RunId,
    TaskId,
)
from domain.value_objects.limits import RunLimits
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolKind, ToolResult
from domain.value_objects.worker import JobRequirements
from infrastructure.database.engine import (
    create_database_engine,
    create_session_factory,
    to_async_url,
)
from infrastructure.database.models import Base

# Pinned rather than ``latest``: partial unique indexes and advisory locks must
# be exercised against the version the platform actually deploys.
POSTGRES_IMAGE = os.environ.get("TEST_POSTGRES_IMAGE", "postgres:16-alpine")


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A usable PostgreSQL, from the environment or from a throwaway container."""
    provided = os.environ.get("TEST_DATABASE_URL")
    if provided:
        yield to_async_url(provided)
        return

    postgres = pytest.importorskip(
        "testcontainers.community.postgres",
        reason="testcontainers is not installed",
    )
    try:
        container = postgres.PostgresContainer(POSTGRES_IMAGE, driver="asyncpg")
        container.start()
    except Exception as exc:  # a missing daemon, a missing image, a refused socket
        pytest.skip(f"no PostgreSQL available for integration tests: {exc}")

    try:
        yield str(container.get_connection_url())
    finally:
        container.stop()


@pytest.fixture
async def database_engine(postgres_url: str) -> AsyncIterator[AsyncEngine]:
    """A fresh schema per test.

    Dropping and recreating is cheap here and buys complete isolation: the
    sequence and idempotency tests assert on "nothing exists yet" states that
    leftovers from a previous test would quietly invalidate.
    """
    engine = create_database_engine(postgres_url, pool_size=2, max_overflow=0)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def session_factory(database_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(database_engine)


# ---------------------------------------------------------------------------
# domain object builders
#
# Exposed as fixtures rather than importable helpers: ``tests`` is not a
# package, so a plain module-level import from one test file to another would
# depend on pytest's sys.path insertion order.
# ---------------------------------------------------------------------------


DIFF = """diff --git a/app/api.py b/app/api.py
--- a/app/api.py
+++ b/app/api.py
@@ -1,2 +1,3 @@
 import fastapi
+# healthcheck
-# todo
"""


@pytest.fixture
def new_run(project: Project, now: datetime) -> Callable[..., Run]:
    def build(
        *,
        objective: str = "expose a /healthz endpoint",
        candidate_count: int = 2,
        idempotency_key: str | None = None,
        limits: RunLimits | None = None,
    ) -> Run:
        return Run.create(
            run_id=RunId.generate(),
            project_id=project.id,
            objective=objective,
            now=now,
            candidate_count=candidate_count,
            limits=limits or RunLimits(max_parallel_candidates=3, max_repair_iterations=2),
            idempotency_key=IdempotencyKey(idempotency_key) if idempotency_key else None,
            metadata={"requested_by": "integration-test"},
        )

    return build


@pytest.fixture
def new_job(project: Project, now: datetime) -> Callable[..., Job]:
    def build(
        *,
        run: Run,
        job_type: JobType = JobType.CODE,
        role: AgentRole | None = AgentRole.CODER,
        candidate_id: CandidateId | None = None,
        idempotency_key: str | None = None,
        priority: Priority = Priority.HIGH,
    ) -> Job:
        return Job.create(
            job_id=JobId.generate(),
            run_id=run.id,
            project_id=project.id,
            job_type=job_type,
            now=now,
            role=role,
            candidate_id=candidate_id,
            priority=priority,
            payload={"prompt_version": "v1", "task_keys": ["t1", "t2"]},
            requirements=JobRequirements(
                role=role or AgentRole.CODER,
                model_id="Qwen3-Coder-30B-A3B",
                estimated_prompt_tokens=12_000,
                requires_tools=True,
            ),
            max_attempts=3,
            idempotency_key=IdempotencyKey(idempotency_key) if idempotency_key else None,
        )

    return build


@pytest.fixture
def new_candidate(now: datetime) -> Callable[..., Candidate]:
    def build(*, run: Run, index: int = 0) -> Candidate:
        return Candidate.create(
            candidate_id=CandidateId.generate(), run_id=run.id, index=index, now=now
        )

    return build


@pytest.fixture
def new_plan(now: datetime) -> Callable[..., Plan]:
    def build(*, run: Run, revision: int = 1) -> Plan:
        return Plan.create(
            plan_id=PlanId.generate(),
            run_id=run.id,
            revision=revision,
            objective=run.objective,
            now=now,
            tasks=[
                PlanTask(
                    id=TaskId.generate(),
                    key="route",
                    title="Add the route",
                    description="Register /healthz on the app",
                    target_paths=("app/api.py",),
                    validation=("pytest -q",),
                ),
                PlanTask(
                    id=TaskId.generate(),
                    key="tests",
                    title="Cover the route",
                    depends_on=("route",),
                ),
            ],
            assumptions=("the app uses FastAPI",),
            constraints=("no new dependency",),
            risk_areas=("routing table",),
            validation_requirements=("pytest -q",),
        )

    return build


@pytest.fixture
def new_review(now: datetime) -> Callable[..., Review]:
    def build(*, run: Run, candidate: Candidate, iteration: int = 1) -> Review:
        return Review(
            id=ReviewId.generate(),
            run_id=run.id,
            candidate_id=candidate.id,
            verdict=ReviewVerdict.FAIL,
            iteration=iteration,
            created_at=now,
            summary="the endpoint is untested",
            findings=(
                ReviewFinding(
                    summary="no test covers /healthz",
                    severity=Severity.MAJOR,
                    file="tests/test_api.py",
                    line=12,
                    repair_instruction="add a test asserting a 200 response",
                ),
                ReviewFinding(summary="docstring missing", severity=Severity.MINOR),
            ),
            metadata={"model": "reviewer-v1"},
        )

    return build


@pytest.fixture
def tool_results() -> tuple[ToolResult, ...]:
    return (
        ToolResult(
            tool="build",
            kind=ToolKind.BUILD,
            command="python -m compileall .",
            exit_code=0,
            stdout="ok",
            duration_ms=1200,
        ),
        ToolResult(
            tool="pytest",
            kind=ToolKind.TEST,
            command="pytest -q",
            exit_code=1,
            stdout="1 failed",
            stderr="assert 0",
            duration_ms=5400,
            truncated=True,
            artifacts=("junit.xml",),
            metadata={"failed": 1},
        ),
    )


@pytest.fixture
def candidate_patch() -> Patch:
    return Patch.from_unified_diff(DIFF, base_revision="deadbeef")


@pytest.fixture
def lease_duration() -> timedelta:
    return timedelta(minutes=5)
