"""HTTP client for the control plane's internal worker API (spec section 19)."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Self

import httpx

__all__ = ["ControlPlaneClient", "ControlPlaneError"]

_log = logging.getLogger(__name__)


class ControlPlaneError(RuntimeError):
    """The control plane refused or could not be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_unknown_worker(self) -> bool:
        """A 404 means the control plane forgot us; re-registering is the fix."""
        return self.status_code == httpx.codes.NOT_FOUND


class ControlPlaneClient:
    """Thin, retry-free transport. Retries belong to the caller's loop."""

    def __init__(
        self,
        *,
        base_url: str,
        service_token: str,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        # Kept on the instance rather than only on the client's default headers:
        # an injected client carries its own defaults, and a token silently
        # dropped that way turns every call into a 401 with nothing to point at.
        self._headers = _auth_headers(service_token)
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds, headers=self._headers
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/internal/workers/register", payload)

    async def heartbeat(self, worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post(f"/internal/workers/{worker_id}/heartbeat", payload)

    async def drain(self, worker_id: str) -> dict[str, Any]:
        return await self._post(f"/internal/workers/{worker_id}/drain", {})

    async def deregister(self, worker_id: str) -> None:
        await self._request("DELETE", f"/internal/workers/{worker_id}")

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request("POST", path, json=payload)
        if not response.content:
            return {}
        body = response.json()
        return body if isinstance(body, dict) else {}

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {**self._headers, **(kwargs.pop("headers", None) or {})}
        try:
            response = await self._client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ControlPlaneError(f"{method} {path} failed: {exc}") from exc
        if response.is_error:
            raise ControlPlaneError(
                f"{method} {path} returned {response.status_code}: {response.text[:400]}",
                status_code=response.status_code,
            )
        return response


def _auth_headers(token: str) -> dict[str, str]:
    """Service authentication. An empty token is allowed only for local runs,
    where the control plane accepts unauthenticated internal calls."""
    return {"Authorization": f"Bearer {token}"} if token else {}
