"""The exposition, which until now existed only as a module nobody imported.

`PlatformMetrics` was declared on the first day and never instantiated: no
recorder was built, no route served it, and `METRICS_ENABLED` gated nothing.
That is not a missing metric — an unfed Prometheus gauge reads **zero**, so a
dashboard wired to this would have reported "no active runs, no registered
workers" with all the confidence of a measurement.
"""

from __future__ import annotations

from dataclasses import replace

import httpx

from interfaces.api.app import create_api


async def test_the_exposition_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


async def test_it_carries_the_instruments_the_platform_declares(
    client: httpx.AsyncClient,
) -> None:
    body = (await client.get("/metrics")).text

    for name in ("active_runs", "queued_jobs", "registered_workers", "healthy_workers"):
        assert name in body, f"{name} is declared but absent from the exposition"


async def test_it_needs_no_service_token(client: httpx.AsyncClient) -> None:
    """Like the probes. It exposes counts and durations, never a payload, an
    identifier or a secret, and a scraper must not hold a credential that also
    opens the internal worker routes."""
    assert (await client.get("/metrics")).status_code == 200


async def test_a_disabled_registry_is_a_404_not_an_empty_200(harness) -> None:
    """A scrape that succeeds and returns nothing is indistinguishable from a
    platform doing nothing, which is the confusion this whole entry is about."""
    disabled = create_api(dependencies=replace(harness.app.state.dependencies, metrics=None))
    transport = httpx.ASGITransport(app=disabled)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        assert (await client.get("/metrics")).status_code == 404
