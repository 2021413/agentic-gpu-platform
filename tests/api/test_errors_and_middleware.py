"""The error mapping and the request-correlation middleware.

The mapping is exercised directly rather than through routes: most domain errors
are raised deep inside an orchestration the API cannot trigger on demand, and a
table is exactly the kind of thing that must be checked as a table.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from tests.api.conftest import Harness

from domain.enums import JobStatus, RunStatus
from domain.exceptions import (
    DomainError,
    EntityNotFoundError,
    IdempotencyConflictError,
    InferenceError,
    InvalidStateTransitionError,
    JobLeaseExpiredError,
    JobNotRetryableError,
    LLMTimeoutError,
    NoCompatibleWorkerError,
    PlanValidationError,
    RunCancelledError,
    RunNotModifiableError,
    StructuredOutputError,
    ToolExecutionError,
    WorkerUnavailableError,
    WorkspaceError,
)
from domain.value_objects.identifiers import JobId, RunId, WorkerId
from interfaces.api.errors import STATUS_BY_CODE, status_for_code


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (EntityNotFoundError("Run", RunId.generate()), 404),
        (InvalidStateTransitionError("Run", RunStatus.COMPLETED, RunStatus.CODING), 409),
        (RunNotModifiableError(RunId.generate(), RunStatus.COMPLETED), 409),
        (RunCancelledError(RunId.generate()), 409),
        (IdempotencyConflictError("key"), 409),
        (JobLeaseExpiredError(JobId.generate()), 409),
        (JobNotRetryableError(JobId.generate(), JobStatus.DEAD, 3), 409),
        (WorkerUnavailableError(WorkerId.generate()), 409),
        (PlanValidationError("cycle between tasks"), 422),
        (StructuredOutputError("bad json", schema="plan"), 422),
        (LLMTimeoutError(30.0), 504),
        (InferenceError("upstream refused", status_code=500), 502),
        (NoCompatibleWorkerError(), 503),
        (ToolExecutionError("pytest", "binary missing"), 500),
        (WorkspaceError("worktree is locked"), 500),
        (DomainError("something unclassified"), 500),
    ],
)
def test_every_domain_error_has_a_deliberate_status(
    error: DomainError, expected_status: int
) -> None:
    assert status_for_code(error.code) == expected_status


def test_an_unknown_code_is_a_server_fault_not_a_client_one() -> None:
    """A code the boundary was never taught about is our omission, so 5xx."""
    assert "invented_code" not in STATUS_BY_CODE
    assert status_for_code("invented_code") == 500


async def test_a_generated_request_id_is_returned(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.headers["x-request-id"]
    assert float(response.headers["x-response-time-ms"]) >= 0


async def test_a_supplied_request_id_is_echoed_back(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "trace-42"})

    assert response.headers["x-request-id"] == "trace-42"


async def test_a_forged_request_id_is_discarded(client: httpx.AsyncClient) -> None:
    """Control characters in a header end up in log lines; they are dropped."""
    response = await client.get("/health", headers={"X-Request-ID": "linebreak"})

    assert response.headers["x-request-id"] != "linebreak"


async def test_the_access_log_carries_the_request_id(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="interfaces.api.access"):
        await client.get("/health", headers={"X-Request-ID": "trace-77"})

    record = next(r for r in caplog.records if r.name == "interfaces.api.access")
    assert record.request_id == "trace-77"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.duration_ms >= 0


async def test_an_unknown_route_is_still_problem_json(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/nope")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "not_found"


async def test_an_unexpected_failure_never_leaks_its_message(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """A 500 body is deliberately opaque; the request id points at the logs.

    Starlette renders an unhandled exception outside every user middleware, so
    this response carries no ``X-Request-ID`` header. The correlation id still
    reaches the caller in the body and the access log, which is what makes the
    two joinable afterwards.
    """

    async def explode() -> None:
        raise RuntimeError("connection string postgres://user:secret@db/app")

    harness.app.router.get("/boom")(explode)

    transport = httpx.ASGITransport(app=harness.app, raise_app_exceptions=False)
    with caplog.at_level(logging.INFO, logger="interfaces.api.access"):
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            response = await client.get("/boom", headers={"X-Request-ID": "trace-boom"})

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert "secret" not in response.text
    assert body["request_id"] == "trace-boom"

    logged = next(r for r in caplog.records if getattr(r, "request_id", None) == "trace-boom")
    assert logged.status_code == 500
