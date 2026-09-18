"""The inference port (spec section 18).

Deliberately free of any provider vocabulary: no ``vllm``, no ``openai``, no
``qwen``. Swapping the engine must not touch a single line of orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from domain.value_objects.llm import CompletionRequest, CompletionResult, ModelInfo
from domain.value_objects.worker import WorkerEndpoint

__all__ = ["LLMProvider", "LLMProviderFactory"]


@runtime_checkable
class LLMProvider(Protocol):
    """Turns a request into a completion, against one concrete worker."""

    @property
    def model_info(self) -> ModelInfo:
        """Metadata about the served model."""
        ...

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion.

        Must raise ``LLMTimeoutError`` on deadline breach and must honour
        ``asyncio`` cancellation so a cancelled run stops burning GPU time.
        """
        ...

    def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Yield incremental content chunks.

        Declared as a plain ``def`` returning an async iterator, like
        ``EventBus.subscribe``: an ``async def`` here would be ambiguous, since a
        coroutine returning an iterator and an async generator share that
        annotation but are consumed differently (``async for await f()`` versus
        ``async for f()``). Implementations are async generators.

        Streaming carries visible output only; hidden reasoning is never
        forwarded to clients (spec section 21).
        """
        ...

    async def health(self) -> bool:
        """Cheap liveness probe used before scheduling onto a worker."""
        ...


@runtime_checkable
class LLMProviderFactory(Protocol):
    """Builds a provider bound to a specific worker endpoint.

    The orchestrator holds workers, not providers: a provider is created per
    job, which is what lets workers come and go mid-run.
    """

    def for_endpoint(self, endpoint: WorkerEndpoint, *, model_id: str) -> LLMProvider: ...
