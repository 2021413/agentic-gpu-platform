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

__all__ = ["InProcessInferenceProbe", "InferenceProbe"]

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

    async def served_model_id(self) -> str | None:
        """The name the engine answers to, or ``None`` if it does not say.

        The control plane sends this verbatim as the ``model`` field, and vLLM
        refuses any name it does not serve. The value is decided in three
        places — the repository id, SERVED_MODEL_NAME, and whatever the agent
        was configured with — so asking the engine is the only way to be right.
        """
        card = await self._first_model_card()
        served = card.get("id") if card else None
        return served if isinstance(served, str) and served else None

    async def _first_model_card(self) -> dict[str, object] | None:
        try:
            response = await self._client.get("/v1/models")
        except httpx.HTTPError:
            return None
        if not response.is_success:
            return None
        try:
            models = response.json().get("data") or []
        except ValueError:
            return None
        cards = [card for card in models if isinstance(card, dict)]
        return cards[0] if cards else None

    async def served_context_length(self) -> int | None:
        """The context window the engine is actually serving, or ``None``.

        vLLM reports ``max_model_len`` on its model card, and that is the only
        number that matters to the scheduler: the model's native context is
        what it *could* serve, while MAX_MODEL_LEN is what this process *will*
        accept. Declaring the former made the control plane believe in sixteen
        times the room it had.

        ``None`` means the engine did not say. That must stay distinguishable
        from a number, because guessing here is how the two sides drifted apart
        in the first place.
        """
        try:
            response = await self._client.get("/v1/models")
        except httpx.HTTPError:
            return None
        if not response.is_success:
            return None
        try:
            models = response.json().get("data") or []
        except ValueError:
            return None
        lengths = [
            int(card["max_model_len"])
            for card in models
            if isinstance(card, dict) and isinstance(card.get("max_model_len"), int)
        ]
        # The smallest, when several are served: a job routed to this endpoint
        # may land on any of them, so the honest capacity is the narrowest.
        return min(lengths) if lengths else None

    async def active_requests(self) -> int | None:
        """Best-effort occupancy. ``None`` when the engine does not report it,
        in which case the control plane relies on its own accounting."""
        return None


class InProcessInferenceProbe(InferenceProbe):
    """For a provider that has no server to poll.

    The fake provider runs inside the control plane, so a worker backed by it
    exposes no inference endpoint. Waiting for one is not caution, it is a
    deadlock: the agent sat out its whole startup timeout and never registered.

    It reports ready, and reports no served context length — it serves nothing,
    so it has nothing to say about a window, and the declared value stands.
    """

    def __init__(self) -> None:
        super().__init__(base_url="http://in-process.invalid")

    async def is_healthy(self) -> bool:
        return True

    async def wait_until_ready(
        self, *, timeout_seconds: float = 900.0, poll_seconds: float = 3.0
    ) -> bool:
        return True

    async def served_context_length(self) -> int | None:
        return None

    async def active_requests(self) -> int | None:
        return None
