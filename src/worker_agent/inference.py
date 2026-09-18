"""Probing the local inference server.

The worker must not advertise itself as READY before its engine can actually
serve a request: registering early is how a fresh GPU ends up handed a job it
drops on the floor.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Self

import httpx

__all__ = ["InferenceProbe"]

_log = logging.getLogger(__name__)


class InferenceProbe:
    """Liveness of an OpenAI-compatible inference server."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds, headers=headers
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

    async def is_healthy(self) -> bool:
        """True when the engine answers. Probes ``/health`` then ``/v1/models``,
        because not every OpenAI-compatible server exposes the former."""
        for path in ("/health", "/v1/models"):
            try:
                response = await self._client.get(path)
            except httpx.HTTPError:
                continue
            if response.is_success:
                return True
        return False

    async def wait_until_ready(
        self, *, timeout_seconds: float = 900.0, poll_seconds: float = 3.0
    ) -> bool:
        """Block until the engine answers or the deadline passes.

        Loading tens of gigabytes of weights takes minutes; the generous default
        reflects that rather than pretending startup is instant.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            if await self.is_healthy():
                return True
            await asyncio.sleep(poll_seconds)
        return False

    async def active_requests(self) -> int | None:
        """Best-effort occupancy. ``None`` when the engine does not report it,
        in which case the control plane relies on its own accounting."""
        return None
