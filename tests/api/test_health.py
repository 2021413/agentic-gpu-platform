"""Liveness and readiness are not the same endpoint, and behave differently."""

from __future__ import annotations

import httpx
from tests.api.conftest import Harness


async def test_liveness_ignores_dependencies(client: httpx.AsyncClient, harness: Harness) -> None:
    """A dead database must not get this container restarted."""
    harness.probe.healthy = False

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_readiness_reports_every_dependency(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert {dependency["name"] for dependency in body["dependencies"]} == {"database", "redis"}


async def test_readiness_fails_when_a_dependency_is_down(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    """503 takes the instance out of the load balancer without killing it."""
    harness.probe.healthy = False

    response = await client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert any(dependency["detail"] == "connection refused" for dependency in body["dependencies"])
