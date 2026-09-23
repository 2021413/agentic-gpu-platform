"""The OpenAI-compatible adapter, against a simulated server.

Everything here runs on ``httpx.MockTransport``: no network, no GPU, no vLLM.
What is asserted is the contract with the outside world — the exact payload sent
and the exact domain objects produced — because that contract is the one thing
integration tests on real hardware cannot iterate on quickly.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
import pytest

from domain.exceptions import InferenceError, LLMTimeoutError
from domain.ports.llm_provider import LLMProvider, LLMProviderFactory
from domain.value_objects.llm import (
    ChatMessage,
    ChatRole,
    CompletionRequest,
    FinishReason,
    ModelInfo,
    ToolSpec,
)
from domain.value_objects.worker import WorkerEndpoint
from infrastructure.llm import openai_compatible
from infrastructure.llm.openai_compatible import (
    HttpLLMProviderFactory,
    OpenAICompatibleLLMProvider,
    OpenAICompatibleSettings,
)

Handler = Callable[[httpx.Request], httpx.Response | Coroutine[Any, Any, httpx.Response]]

MODEL = "configured-model-id"
BASE_URL = "http://worker.invalid:8000"


def make_provider(
    handler: Handler,
    *,
    settings: OpenAICompatibleSettings | None = None,
    model_id: str = MODEL,
    supports_json_schema: bool = True,
) -> OpenAICompatibleLLMProvider:
    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return OpenAICompatibleLLMProvider(
        client=client,
        model_info=ModelInfo(
            model_id=model_id,
            context_length=8_192,
            supports_json_schema=supports_json_schema,
        ),
        settings=settings or OpenAICompatibleSettings(),
    )


def chat_response(
    *,
    content: str | None = "ok",
    finish_reason: str = "stop",
    extra_message: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message.update(extra_message or {})
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage if usage is not None else {"prompt_tokens": 11, "completion_tokens": 5},
        },
    )


def simple_request(**kwargs: Any) -> CompletionRequest:
    return CompletionRequest(
        messages=(ChatMessage.system("be terse"), ChatMessage.user("hello")), **kwargs
    )


class Recorder:
    """Captures the request bodies the adapter sends."""

    def __init__(self, response_factory: Callable[[], httpx.Response] = chat_response) -> None:
        self.requests: list[httpx.Request] = []
        self.payloads: list[dict[str, Any]] = []
        self._response_factory = response_factory

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.payloads.append(json.loads(request.content) if request.content else {})
        return self._response_factory()


# -- request mapping -------------------------------------------------------
async def test_request_is_mapped_onto_the_chat_completions_payload() -> None:
    recorder = Recorder()
    provider = make_provider(recorder)

    await provider.complete(
        CompletionRequest(
            messages=(
                ChatMessage.system("be terse"),
                ChatMessage.user("hello"),
                ChatMessage(ChatRole.TOOL, "42", name="answer", tool_call_id="call-1"),
            ),
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            stop=("STOP",),
            seed=7,
            correlation_id="run-42",
        )
    )

    payload = recorder.payloads[0]
    assert recorder.requests[0].url.path == "/v1/chat/completions"
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 256
    assert payload["stop"] == ["STOP"]
    assert payload["seed"] == 7
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "42", "name": "answer", "tool_call_id": "call-1"},
    ]
    assert recorder.requests[0].headers["X-Request-Id"] == "run-42"


async def test_the_model_comes_from_configuration_not_from_the_adapter() -> None:
    recorder = Recorder()
    provider = make_provider(recorder, model_id="some-other-model")

    await provider.complete(simple_request())
    await provider.complete(simple_request(model="pinned-by-the-caller"))

    assert recorder.payloads[0]["model"] == "some-other-model"
    assert recorder.payloads[1]["model"] == "pinned-by-the-caller"


def test_no_model_name_is_hard_coded_in_the_adapter() -> None:
    """Spec section 18: abstractions must not be named after one model."""
    source = Path(openai_compatible.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("qwen", "gpt-", "llama", "mistral"):
        assert forbidden not in source


async def test_api_key_and_extra_headers_are_sent() -> None:
    recorder = Recorder()
    provider = make_provider(
        recorder,
        settings=OpenAICompatibleSettings(api_key="secret", extra_headers={"X-Tenant": "acme"}),
    )

    await provider.complete(simple_request())

    headers = recorder.requests[0].headers
    assert headers["Authorization"] == "Bearer secret"
    assert headers["X-Tenant"] == "acme"


async def test_json_schema_becomes_a_response_format() -> None:
    recorder = Recorder()
    provider = make_provider(recorder)
    schema = {"title": "PlannerOutput", "type": "object", "properties": {}}

    await provider.complete(simple_request(json_schema=schema))

    response_format = recorder.payloads[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "PlannerOutput"
    assert response_format["json_schema"]["schema"] == schema
    assert response_format["json_schema"]["strict"] is True


async def test_json_schema_degrades_to_json_mode_when_the_worker_cannot_constrain() -> None:
    recorder = Recorder()
    provider = make_provider(recorder, supports_json_schema=False)

    await provider.complete(simple_request(json_schema={"title": "PlannerOutput"}))

    assert recorder.payloads[0]["response_format"] == {"type": "json_object"}


async def test_tools_are_advertised_as_functions() -> None:
    recorder = Recorder()
    provider = make_provider(recorder)
    tool = ToolSpec(
        name="read_file",
        description="read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    await provider.complete(simple_request(tools=(tool,)))

    payload = recorder.payloads[0]
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


async def test_no_response_format_or_tools_when_not_requested() -> None:
    recorder = Recorder()
    provider = make_provider(recorder)

    await provider.complete(simple_request())

    assert "response_format" not in recorder.payloads[0]
    assert "tools" not in recorder.payloads[0]


# -- response mapping ------------------------------------------------------
async def test_response_is_mapped_onto_a_completion_result() -> None:
    provider = make_provider(lambda _r: chat_response(content="the answer"))

    result = await provider.complete(simple_request(correlation_id="run-7"))

    assert result.content == "the answer"
    assert result.model == MODEL
    assert result.finish_reason is FinishReason.STOP
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 16
    assert result.latency_ms >= 0
    assert result.correlation_id == "run-7"
    assert result.truncated is False


async def test_truncated_answers_are_reported_as_such() -> None:
    provider = make_provider(lambda _r: chat_response(content='{"objec', finish_reason="length"))

    result = await provider.complete(simple_request())

    assert result.finish_reason is FinishReason.LENGTH
    assert result.truncated is True


async def test_tool_calls_are_returned_with_raw_arguments() -> None:
    provider = make_provider(
        lambda _r: chat_response(
            content=None,
            finish_reason="tool_calls",
            extra_message={
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "grep", "arguments": {"pattern": "x"}},
                    },
                ]
            },
        )
    )

    result = await provider.complete(simple_request())

    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert result.content == ""
    assert [(c.id, c.name) for c in result.tool_calls] == [
        ("call-1", "read_file"),
        ("call-2", "grep"),
    ]
    assert json.loads(result.tool_calls[1].arguments) == {"pattern": "x"}


async def test_hidden_reasoning_never_reaches_the_result() -> None:
    """Spec section 21: reasoning fields are dropped, not concatenated."""
    provider = make_provider(
        lambda _r: chat_response(
            content="visible answer",
            extra_message={"reasoning_content": "first I will secretly consider..."},
        )
    )

    result = await provider.complete(simple_request())

    assert result.content == "visible answer"
    assert "secretly" not in result.content


async def test_content_parts_are_flattened() -> None:
    provider = make_provider(
        lambda _r: httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "part one "},
                                {"type": "text", "text": "part two"},
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    result = await provider.complete(simple_request())

    assert result.content == "part one part two"
    assert result.usage.total_tokens == 0


async def test_unknown_finish_reason_is_reported_as_an_error() -> None:
    provider = make_provider(lambda _r: chat_response(finish_reason="content_filter"))

    result = await provider.complete(simple_request())

    assert result.finish_reason is FinishReason.ERROR


# -- failure mapping -------------------------------------------------------
async def test_server_error_becomes_a_domain_error() -> None:
    provider = make_provider(lambda _r: httpx.Response(500, text="engine died"))

    with pytest.raises(InferenceError) as excinfo:
        await provider.complete(simple_request())

    assert excinfo.value.status_code == 500
    assert excinfo.value.is_retryable is True
    assert "engine died" in excinfo.value.message


async def test_client_error_is_not_retryable() -> None:
    provider = make_provider(lambda _r: httpx.Response(400, text="bad schema"))

    with pytest.raises(InferenceError) as excinfo:
        await provider.complete(simple_request())

    assert excinfo.value.is_retryable is False


async def test_connection_failure_never_surfaces_as_an_httpx_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = make_provider(refuse)

    with pytest.raises(InferenceError) as excinfo:
        await provider.complete(simple_request())

    assert excinfo.value.status_code is None
    assert excinfo.value.is_retryable is True


async def test_non_json_body_becomes_a_domain_error() -> None:
    provider = make_provider(lambda _r: httpx.Response(200, text="<html>proxy error</html>"))

    with pytest.raises(InferenceError):
        await provider.complete(simple_request())


async def test_answer_without_choices_becomes_a_domain_error() -> None:
    provider = make_provider(lambda _r: httpx.Response(200, json={"model": MODEL, "choices": []}))

    with pytest.raises(InferenceError):
        await provider.complete(simple_request())


async def test_transport_timeout_becomes_an_llm_timeout() -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = make_provider(time_out)

    with pytest.raises(LLMTimeoutError):
        await provider.complete(simple_request(timeout_seconds=30))


async def test_deadline_breach_becomes_an_llm_timeout() -> None:
    async def never_answers(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return chat_response()

    provider = make_provider(never_answers)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await provider.complete(simple_request(timeout_seconds=0.05))

    assert excinfo.value.timeout_seconds == 0.05
    assert excinfo.value.details["model"] == MODEL


async def test_cancellation_reaches_the_transport() -> None:
    """A cancelled run must stop burning GPU time, not just stop waiting."""
    started = asyncio.Event()
    aborted = asyncio.Event()

    async def long_generation(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            aborted.set()
            raise
        return chat_response()  # pragma: no cover - the sleep is always cancelled

    provider = make_provider(long_generation)
    task = asyncio.create_task(provider.complete(simple_request(timeout_seconds=30)))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert aborted.is_set()


# -- health ----------------------------------------------------------------
async def test_health_uses_the_health_endpoint() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    provider = make_provider(handler)

    assert await provider.health() is True
    assert seen == ["/health"]


async def test_health_falls_back_to_the_model_listing() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": []})

    provider = make_provider(handler)

    assert await provider.health() is True
    assert seen == ["/health", "/v1/models"]


async def test_health_is_false_when_the_worker_is_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = make_provider(refuse)

    assert await provider.health() is False


async def test_health_is_false_when_the_worker_reports_an_error() -> None:
    provider = make_provider(lambda _r: httpx.Response(503))

    assert await provider.health() is False


# -- factory ---------------------------------------------------------------
def build_factory() -> tuple[HttpLLMProviderFactory, list[str]]:
    created: list[str] = []

    def client_factory(base_url: str) -> httpx.AsyncClient:
        created.append(base_url)
        return httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(lambda _r: chat_response())
        )

    return HttpLLMProviderFactory(
        OpenAICompatibleSettings(context_length=4_096), client_factory=client_factory
    ), created


async def test_factory_pools_one_client_per_endpoint() -> None:
    factory, created = build_factory()
    endpoint = WorkerEndpoint("http://worker-a:8000")

    first = factory.for_endpoint(endpoint, model_id=MODEL)
    second = factory.for_endpoint(WorkerEndpoint("http://worker-a:8000/"), model_id="other")
    third = factory.for_endpoint(WorkerEndpoint("http://worker-b:8000"), model_id=MODEL)

    assert created == ["http://worker-a:8000", "http://worker-b:8000"]
    assert first is not second  # providers are cheap, per-job objects
    assert first.model_info.model_id == MODEL
    assert second.model_info.model_id == "other"
    assert third.model_info.context_length == 4_096
    await factory.aclose()


async def test_factory_conforms_to_the_ports() -> None:
    factory, _ = build_factory()
    typed_factory: LLMProviderFactory = factory
    provider: LLMProvider = typed_factory.for_endpoint(
        WorkerEndpoint("http://worker-a:8000"), model_id=MODEL
    )

    assert isinstance(provider, LLMProvider)
    assert isinstance(factory, LLMProviderFactory)
    await factory.aclose()


async def test_closing_the_factory_closes_every_pooled_client() -> None:
    clients: list[httpx.AsyncClient] = []

    def client_factory(base_url: str) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(lambda _r: chat_response())
        )
        clients.append(client)
        return client

    async with HttpLLMProviderFactory(client_factory=client_factory) as factory:
        factory.for_endpoint(WorkerEndpoint("http://worker-a:8000"), model_id=MODEL)
        factory.for_endpoint(WorkerEndpoint("http://worker-b:8000"), model_id=MODEL)

    assert [client.is_closed for client in clients] == [True, True]


async def test_a_closed_factory_refuses_to_build_providers() -> None:
    factory, _ = build_factory()
    await factory.aclose()

    with pytest.raises(RuntimeError):
        factory.for_endpoint(WorkerEndpoint("http://worker-a:8000"), model_id=MODEL)


def test_factory_requires_a_model_id() -> None:
    factory, _ = build_factory()

    with pytest.raises(ValueError, match="model_id"):
        factory.for_endpoint(WorkerEndpoint("http://worker-a:8000"), model_id="")


# -- serverless endpoints (Modal Servers) ----------------------------------
#
# A Modal Server does not queue: with no container running, its proxy answers
# 503 immediately and starts one in response to that same request. Every test
# below exists because getting this wrong is not a crash but a bill — a cold
# start paid on every idle cycle and attributed to a job that failed.

SERVERLESS = OpenAICompatibleSettings(
    scale_to_zero=True,
    cold_start_max_wait_seconds=1.0,
    cold_start_poll_seconds=0.01,
)


def cold_then(cold: int, served: Handler) -> Handler:
    """A server that answers 503 ``cold`` times, then hands over to ``served``."""
    remaining = {"n": cold}

    def handler(request: httpx.Request) -> httpx.Response:
        if remaining["n"] > 0:
            remaining["n"] -= 1
            return httpx.Response(503, text="no containers available")
        result = served(request)
        assert isinstance(result, httpx.Response)
        return result

    return handler


async def test_a_cold_serverless_endpoint_is_waited_for() -> None:
    recorder = Recorder()
    provider = make_provider(cold_then(2, recorder), settings=SERVERLESS)

    result = await provider.complete(simple_request())

    assert result.content == "ok"
    assert len(recorder.requests) == 1, "only the successful attempt reaches the engine"


async def test_the_retried_request_carries_its_body_again() -> None:
    """An httpx.Request cannot be sent twice: its stream is consumed.

    Reusing one would send an empty body to the container that finally booted,
    and vLLM would answer 400 to a request the caller believes it sent.
    """
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        if len(seen) < 3:
            return httpx.Response(503, text="no containers available")
        return chat_response()

    provider = make_provider(handler, settings=SERVERLESS)
    await provider.complete(simple_request())

    assert len(seen) == 3
    assert all(body and json.loads(body)["messages"] for body in seen)


async def test_a_serverless_endpoint_gives_up_when_the_budget_runs_out() -> None:
    provider = make_provider(
        lambda _r: httpx.Response(503, text="no containers available"),
        settings=OpenAICompatibleSettings(
            scale_to_zero=True,
            cold_start_max_wait_seconds=0.05,
            cold_start_poll_seconds=0.01,
        ),
    )

    with pytest.raises(InferenceError) as caught:
        await provider.complete(simple_request())

    assert caught.value.status_code == 503
    assert "no worker became available" in str(caught.value)


async def test_503_is_a_plain_failure_on_a_dedicated_endpoint() -> None:
    """A RunPod Pod that answers 503 is broken, and waiting would hide it."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="engine unavailable")

    provider = make_provider(handler)

    with pytest.raises(InferenceError) as caught:
        await provider.complete(simple_request())

    assert calls["n"] == 1, "no retry without scale_to_zero"
    assert caught.value.status_code == 503


async def test_health_reports_a_scaled_to_zero_endpoint_as_healthy() -> None:
    """Idle is the resting state of a serverless worker, not a fault.

    Reporting it unhealthy would have the registry evict the only worker there
    is, every time nobody used it for a minute.
    """
    provider = make_provider(
        lambda _r: httpx.Response(503, text="no containers available"), settings=SERVERLESS
    )

    assert await provider.health() is True


async def test_health_reports_503_as_unhealthy_on_a_dedicated_endpoint() -> None:
    provider = make_provider(lambda _r: httpx.Response(503, text="engine unavailable"))

    assert await provider.health() is False


async def test_a_cold_serverless_endpoint_is_waited_for_when_streaming() -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="no containers available")
        body = (
            'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    provider = make_provider(handler, settings=SERVERLESS)

    chunks = [chunk async for chunk in provider.stream(simple_request())]

    assert "".join(chunks) == "hello"
    assert attempts["n"] == 3
