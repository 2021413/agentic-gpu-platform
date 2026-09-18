"""Incremental SSE parsing.

Streaming is where a leak would happen: reasoning deltas, tool-call deltas and
protocol noise all arrive on the same channel as the text a client is watching.
These tests pin down what the adapter forwards — and, just as importantly, what
it drops (spec section 21).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest

from domain.exceptions import InferenceError, LLMTimeoutError
from domain.value_objects.llm import ChatMessage, CompletionRequest, ModelInfo
from infrastructure.llm.openai_compatible import (
    OpenAICompatibleLLMProvider,
    OpenAICompatibleSettings,
    is_stream_end,
    parse_sse_line,
)

MODEL = "configured-model-id"


def delta_event(content: str | None = None, **extra: Any) -> str:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    delta.update(extra)
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"


class TrackedStream(httpx.AsyncByteStream):
    """A response body that records whether the client disconnected."""

    def __init__(self, chunks: Sequence[bytes], *, then_hang: bool = False) -> None:
        self._chunks = chunks
        self._then_hang = then_hang
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._then_hang:
            await asyncio.sleep(30)  # pragma: no cover - always cancelled or closed

    async def aclose(self) -> None:
        self.closed = True


def make_streaming_provider(
    stream: httpx.AsyncByteStream,
    *,
    status_code: int = 200,
) -> tuple[OpenAICompatibleLLMProvider, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            status_code,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    client = httpx.AsyncClient(
        base_url="http://worker.invalid:8000", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        client=client,
        model_info=ModelInfo(model_id=MODEL, context_length=8_192),
        settings=OpenAICompatibleSettings(),
    )
    return provider, payloads


def request() -> CompletionRequest:
    return CompletionRequest(messages=(ChatMessage.user("write a haiku"),))


async def collect(provider: OpenAICompatibleLLMProvider) -> list[str]:
    return [chunk async for chunk in provider.stream(request())]


# -- line level ------------------------------------------------------------
@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        ": keep-alive",
        "event: message",
        "data: [DONE]",
        "data:  ",
    ],
)
def test_protocol_noise_yields_no_content(line: str) -> None:
    assert parse_sse_line(line) is None


def test_reasoning_deltas_are_dropped() -> None:
    assert parse_sse_line(delta_event(reasoning_content="thinking hard").strip()) is None


def test_tool_call_deltas_carry_no_visible_content() -> None:
    event = delta_event(tool_calls=[{"index": 0, "function": {"name": "read_file"}}])
    assert parse_sse_line(event.strip()) is None


def test_a_usage_only_terminal_chunk_yields_nothing() -> None:
    event = "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 3}})
    assert parse_sse_line(event) is None


def test_the_sentinel_is_recognised_whatever_the_spacing() -> None:
    assert is_stream_end("data: [DONE]") is True
    assert is_stream_end("data:[DONE]  ") is True
    assert is_stream_end("data: {}") is False
    assert is_stream_end(": ping") is False


def test_an_error_event_raises() -> None:
    event = "data: " + json.dumps({"error": {"message": "engine out of memory"}})
    with pytest.raises(InferenceError, match="mid-stream"):
        parse_sse_line(event)


# -- stream level ----------------------------------------------------------
async def test_the_stream_is_an_async_generator_not_a_coroutine() -> None:
    """The port is a plain ``def``: ``async for`` must work without awaiting."""
    provider, _ = make_streaming_provider(TrackedStream([b"data: [DONE]\n\n"]))

    stream = provider.stream(request())

    assert inspect.isasyncgen(stream)
    assert [chunk async for chunk in stream] == []


async def test_only_visible_content_is_streamed() -> None:
    body = "".join(
        (
            ": ping\n\n",
            delta_event(reasoning_content="the user probably wants..."),
            delta_event("An old "),
            delta_event(""),
            delta_event("silent pond"),
            delta_event(None, tool_calls=[{"index": 0, "id": "call-1"}]),
            "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 9}}) + "\n\n",
            "data: [DONE]\n\n",
        )
    )
    provider, payloads = make_streaming_provider(TrackedStream([body.encode()]))

    chunks = await collect(provider)

    assert chunks == ["An old ", "silent pond"]
    assert payloads[0]["stream"] is True
    assert payloads[0]["stream_options"] == {"include_usage": True}


async def test_events_split_across_network_chunks_are_reassembled() -> None:
    body = (delta_event("hello ") + delta_event("world") + "data: [DONE]\n\n").encode()
    # Split at deliberately awkward offsets, mid-JSON and mid-line.
    pieces = [body[i : i + 7] for i in range(0, len(body), 7)]
    provider, _ = make_streaming_provider(TrackedStream(pieces))

    assert await collect(provider) == ["hello ", "world"]


async def test_the_stream_stops_at_the_done_sentinel() -> None:
    """Anything a server keeps sending after [DONE] is not part of the answer."""
    body = (delta_event("kept") + "data: [DONE]\n\n" + delta_event("after the sentinel")).encode()
    stream = TrackedStream([body])
    provider, _ = make_streaming_provider(stream)

    assert await collect(provider) == ["kept"]
    assert stream.closed is True


async def test_an_error_status_on_open_becomes_a_domain_error() -> None:
    provider, _ = make_streaming_provider(
        TrackedStream([b'{"error": "no capacity"}']), status_code=503
    )

    with pytest.raises(InferenceError) as excinfo:
        await collect(provider)

    assert excinfo.value.status_code == 503
    assert excinfo.value.is_retryable is True


async def test_a_stalled_stream_becomes_an_llm_timeout() -> None:
    class StallingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield delta_event("first").encode()
            raise httpx.ReadTimeout("stalled")

    provider, _ = make_streaming_provider(StallingStream())

    with pytest.raises(LLMTimeoutError):
        await collect(provider)


async def test_abandoning_the_stream_disconnects_from_the_server() -> None:
    """Closing the iterator must close the response: that is what aborts the GPU."""
    stream = TrackedStream([delta_event("first chunk").encode()], then_hang=True)
    provider, _ = make_streaming_provider(stream)

    iterator = provider.stream(request())
    first = await anext(iterator)
    await iterator.aclose()

    assert first == "first chunk"
    assert stream.closed is True


async def test_cancelling_the_consumer_disconnects_from_the_server() -> None:
    stream = TrackedStream([delta_event("first chunk").encode()], then_hang=True)
    provider, _ = make_streaming_provider(stream)
    started = asyncio.Event()

    async def consume() -> None:
        async for _chunk in provider.stream(request()):
            started.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True
