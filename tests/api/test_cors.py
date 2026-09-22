"""Browser access to the control plane (spec section 20).

`cors_allow_origins` and `allowed_origins` were both defined and neither was
ever read, so a browser could not call this API at all: a viewer served from
its own origin got its preflight refused, and every request failed before it
reached a route. The same shape as several other settings in this codebase —
present, computed, and wired to nothing.

Closed by default: an API that answers any origin with credentials is an API
anybody's page can drive on a logged-in user's behalf.
"""

from __future__ import annotations

import httpx
import pytest

from interfaces.api.app import create_api

VIEWER = "http://localhost:5173"


def app_allowing(*origins: str) -> httpx.AsyncClient:
    app = create_api(allowed_origins=origins, include_internal_api=False)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


async def test_a_configured_origin_is_allowed() -> None:
    async with app_allowing(VIEWER) as client:
        response = await client.get("/health", headers={"origin": VIEWER})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == VIEWER


async def test_a_preflight_from_a_configured_origin_succeeds() -> None:
    """The request a browser actually sends first."""
    async with app_allowing(VIEWER) as client:
        response = await client.request(
            "OPTIONS",
            "/v1/projects",
            headers={
                "origin": VIEWER,
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )

    assert response.status_code in (200, 204), response.text
    assert response.headers.get("access-control-allow-origin") == VIEWER
    assert "POST" in (response.headers.get("access-control-allow-methods") or "")


async def test_an_unconfigured_origin_gets_nothing() -> None:
    async with app_allowing(VIEWER) as client:
        response = await client.get("/health", headers={"origin": "http://evil.invalid"})

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("origins", [(), ("",)])
async def test_configuring_none_leaves_the_api_closed(origins: tuple[str, ...]) -> None:
    """The default. A viewer is opt-in, not something you get by accident."""
    async with app_allowing(*origins) as client:
        response = await client.get("/health", headers={"origin": VIEWER})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
