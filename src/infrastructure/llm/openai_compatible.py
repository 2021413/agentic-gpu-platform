"""Adapter for any server speaking the OpenAI chat-completions dialect.

vLLM is the first target, but nothing here knows that: the dialect is the
contract, the endpoint and the model identifier both come from configuration
(spec section 18). No model name is hard-coded anywhere in this module.

Two behaviours deserve their explanation up front:

*   **Cancellation must reach the GPU.** Every request is sent with
    ``stream=True`` and closed in a ``finally``. Buffering the whole body with a
    plain ``post()`` would leave the socket alive while the task unwinds, and an
    inference server only aborts a generation when its client disconnects. A
    cancelled run must stop burning GPU seconds, not merely stop waiting.
*   **Hidden reasoning never leaves this module.** Only ``message.content`` and
    ``delta.content`` are read. Fields such as ``reasoning_content`` — which
    reasoning-tuned models do emit — are deliberately dropped rather than
    concatenated (spec section 21).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import httpx

from domain.exceptions import InferenceError, LLMTimeoutError
from domain.ports.llm_provider import LLMProvider, LLMProviderFactory
from domain.value_objects.llm import (
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    FinishReason,
    ModelInfo,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from domain.value_objects.worker import WorkerEndpoint

__all__ = [
    "HttpLLMProviderFactory",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleSettings",
    "is_stream_end",
    "parse_sse_line",
]

_EMPTY_HEADERS: Mapping[str, str] = MappingProxyType({})

_FINISH_REASONS: Final[Mapping[str, FinishReason]] = MappingProxyType(
    {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.ERROR,
        "error": FinishReason.ERROR,
        "abort": FinishReason.CANCELLED,
    }
)

_SCHEMA_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_BODY_EXCERPT_CHARS: Final = 500
_DONE_SENTINEL: Final = "[DONE]"


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSettings:
    """Everything the adapter needs that is not part of a request.

    Paths are configurable because "OpenAI-compatible" servers disagree on the
    edges: the chat route is universal, the health route is not.
    """

    api_key: str | None = None
    context_length: int = 32_768
    default_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 5.0
    health_timeout_seconds: float = 5.0
    chat_completions_path: str = "/v1/chat/completions"
    models_path: str = "/v1/models"
    health_path: str = "/health"
    supports_tools: bool = True
    supports_json_schema: bool = True
    max_connections: int = 32
    max_keepalive_connections: int = 8
    extra_headers: Mapping[str, str] = field(default_factory=lambda: _EMPTY_HEADERS)

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")


class OpenAICompatibleLLMProvider:
    """``LLMProvider`` over one concrete endpoint.

    The client is injected, not created: connection pooling is the factory's
    job, and a provider is a cheap per-job object.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model_info: ModelInfo,
        settings: OpenAICompatibleSettings | None = None,
    ) -> None:
        self._client = client
        self._model_info = model_info
        self._settings = settings or OpenAICompatibleSettings()

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        payload = self._chat_payload(request, stream=False)
        started = time.perf_counter()
        body = await self._post(payload, request)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._to_result(body, request, latency_ms)

    async def health(self) -> bool:
        """Probe the server without spending a token.

        Tries the plain health route first and falls back to the model listing,
        because not every compatible server exposes ``/health``.
        """
        for path in (self._settings.health_path, self._settings.models_path):
            try:
                response = await self._client.get(
                    path, timeout=self._settings.health_timeout_seconds
                )
            except httpx.HTTPError:
                return False
            if response.status_code < httpx.codes.BAD_REQUEST:
                return True
            if response.status_code != httpx.codes.NOT_FOUND:
                return False
        return False

    # -- request building ------------------------------------------------
    def _model_for(self, request: CompletionRequest) -> str:
        return request.model or self._model_info.model_id

    def _chat_payload(self, request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_for(request),
            "messages": [_message_payload(m) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.json_schema is not None:
            payload["response_format"] = self._response_format(request.json_schema)
        if request.tools:
            payload["tools"] = [_tool_payload(t) for t in request.tools]
            payload["tool_choice"] = "auto"
        if stream:
            # Usage on the terminal chunk: without it, streamed jobs have no
            # token accounting at all.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _response_format(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Constrained decoding when the worker supports it, JSON mode otherwise.

        Degrading to ``json_object`` keeps a weaker worker usable: the parser
        still validates, and an invalid answer still goes through the repair
        loop instead of failing the run on a capability mismatch.
        """
        if not self._model_info.supports_json_schema:
            return {"type": "json_object"}
        raw_name = str(schema.get("title") or "structured_output")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": _SCHEMA_NAME_RE.sub("_", raw_name),
                "schema": dict(schema),
                "strict": True,
            },
        }

    def _headers(self, request: CompletionRequest) -> dict[str, str]:
        headers = dict(self._settings.extra_headers)
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        if request.correlation_id:
            # Correlating a GPU-side log line with a run is otherwise guesswork.
            headers["X-Request-Id"] = request.correlation_id
            headers["X-Correlation-Id"] = request.correlation_id
        return headers

    def _timeout_seconds(self, request: CompletionRequest) -> float:
        return request.timeout_seconds or self._settings.default_timeout_seconds

    def _httpx_timeout(self, request: CompletionRequest) -> httpx.Timeout:
        seconds = self._timeout_seconds(request)
        return httpx.Timeout(
            seconds,
            connect=min(self._settings.connect_timeout_seconds, seconds),
        )

    def _build_request(
        self, payload: Mapping[str, Any], request: CompletionRequest
    ) -> httpx.Request:
        return self._client.build_request(
            "POST",
            self._settings.chat_completions_path,
            json=payload,
            headers=self._headers(request),
            timeout=self._httpx_timeout(request),
        )

    # -- transport -------------------------------------------------------
    async def _post(self, payload: Mapping[str, Any], request: CompletionRequest) -> dict[str, Any]:
        model = self._model_for(request)
        http_request = self._build_request(payload, request)
        try:
            async with asyncio.timeout(self._timeout_seconds(request)):
                response = await self._client.send(http_request, stream=True)
                try:
                    await response.aread()
                finally:
                    await response.aclose()
        except TimeoutError as exc:
            raise LLMTimeoutError(self._timeout_seconds(request), model=model) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(self._timeout_seconds(request), model=model) from exc
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"inference request failed: {type(exc).__name__}", model=model
            ) from exc

        self._raise_for_status(response, model=model)
        return _json_object(response.text, model=model)

    def _raise_for_status(self, response: httpx.Response, *, model: str) -> None:
        if response.status_code < httpx.codes.BAD_REQUEST:
            return
        raise InferenceError(
            f"inference server returned {response.status_code}: "
            f"{response.text[:_BODY_EXCERPT_CHARS]}",
            status_code=response.status_code,
            model=model,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Yield visible deltas, closing the connection whatever happens.

        An async generator, per the port: ``async for chunk in
        provider.stream(request)``. Closing that iterator — or cancelling its
        consumer — disconnects from the server, which is what aborts the
        generation instead of merely ignoring it.

        No ``asyncio.timeout`` wraps the loop: a deadline spanning ``yield``
        statements would cancel the *consumer* at an arbitrary point. The
        transport's read timeout — which measures the gap between chunks, the
        thing that actually indicates a stalled server — is the right tool.
        """
        model = self._model_for(request)
        payload = self._chat_payload(request, stream=True)
        http_request = self._build_request(payload, request)
        try:
            response = await self._client.send(http_request, stream=True)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(self._timeout_seconds(request), model=model) from exc
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"inference stream could not be opened: {type(exc).__name__}", model=model
            ) from exc

        try:
            if response.status_code >= httpx.codes.BAD_REQUEST:
                await response.aread()
                self._raise_for_status(response, model=model)
            async for line in response.aiter_lines():
                if is_stream_end(line):
                    break
                chunk = parse_sse_line(line, model=model)
                if chunk:
                    yield chunk
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(self._timeout_seconds(request), model=model) from exc
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"inference stream failed: {type(exc).__name__}", model=model
            ) from exc
        finally:
            await response.aclose()

    # -- response mapping ------------------------------------------------
    def _to_result(
        self, body: Mapping[str, Any], request: CompletionRequest, latency_ms: int
    ) -> CompletionResult:
        model = self._model_for(request)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise InferenceError("inference response contained no choices", model=model)
        choice = _as_object(choices[0])
        message = _as_object(choice.get("message"))

        return CompletionResult(
            content=_content_text(message.get("content")),
            model=str(body.get("model") or model),
            finish_reason=_finish_reason(choice.get("finish_reason")),
            tool_calls=_tool_calls(message.get("tool_calls")),
            usage=_usage(body.get("usage")),
            latency_ms=latency_ms,
            correlation_id=request.correlation_id,
        )


class HttpLLMProviderFactory:
    """Builds providers per worker endpoint, pooling one HTTP client each.

    A client per *endpoint* (not per job) is what keeps connections warm: TLS
    and TCP setup on every inference call would show up directly in run latency.
    Closing the factory closes every pooled client.
    """

    def __init__(
        self,
        settings: OpenAICompatibleSettings | None = None,
        *,
        client_factory: Callable[[str], httpx.AsyncClient] | None = None,
    ) -> None:
        self._settings = settings or OpenAICompatibleSettings()
        self._client_factory = client_factory or self._default_client
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._closed = False

    def for_endpoint(self, endpoint: WorkerEndpoint, *, model_id: str) -> LLMProvider:
        if self._closed:
            raise RuntimeError("provider factory is closed")
        if not model_id:
            raise ValueError("model_id must not be empty")
        return OpenAICompatibleLLMProvider(
            client=self._client_for(endpoint),
            model_info=ModelInfo(
                model_id=model_id,
                context_length=self._settings.context_length,
                supports_tools=self._settings.supports_tools,
                supports_json_schema=self._settings.supports_json_schema,
            ),
            settings=self._settings,
        )

    def _client_for(self, endpoint: WorkerEndpoint) -> httpx.AsyncClient:
        key = endpoint.url.rstrip("/")
        client = self._clients.get(key)
        if client is None:
            client = self._client_factory(key)
            self._clients[key] = client
        return client

    def _default_client(self, base_url: str) -> httpx.AsyncClient:
        headers = dict(self._settings.extra_headers)
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        return httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(
                self._settings.default_timeout_seconds,
                connect=self._settings.connect_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=self._settings.max_connections,
                max_keepalive_connections=self._settings.max_keepalive_connections,
            ),
        )

    async def aclose(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        self._closed = True
        for client in clients:
            await client.aclose()

    async def __aenter__(self) -> HttpLLMProviderFactory:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


# -- module-level helpers -------------------------------------------------
def parse_sse_line(line: str, *, model: str | None = None) -> str | None:
    """Turn one SSE line into visible content, or ``None`` when there is none.

    Returns ``None`` for keep-alive comments, blank separators, the terminal
    sentinel, tool-call deltas and — deliberately — any reasoning field.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or not stripped.startswith("data:"):
        return None
    data = stripped[len("data:") :].strip()
    if not data or data == _DONE_SENTINEL:
        return None

    event = _json_object(data, model=model)
    if "error" in event:
        raise InferenceError(
            f"inference server reported an error mid-stream: "
            f"{str(event['error'])[:_BODY_EXCERPT_CHARS]}",
            model=model,
        )
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = _as_object(choices[0])
    delta = _as_object(choice.get("delta"))
    return _content_text(delta.get("content")) or None


def is_stream_end(line: str) -> bool:
    """Whether this line is the terminal sentinel.

    Stopping on it — rather than waiting for the body to end — is what releases
    the connection as soon as the answer is complete.
    """
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return False
    return stripped[len("data:") :].strip() == _DONE_SENTINEL


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters) or {"type": "object", "properties": {}},
        },
    }


def _json_object(text: str, *, model: str | None = None) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise InferenceError(
            f"inference server returned a body that is not JSON: {text[:_BODY_EXCERPT_CHARS]}",
            model=model,
        ) from exc
    if not isinstance(parsed, dict):
        raise InferenceError(
            "inference server returned a JSON value that is not an object", model=model
        )
    return parsed


def _as_object(value: object) -> dict[str, Any]:
    """A JSON field that should be an object, or an empty one.

    Servers disagree on which fields they omit, send as ``null`` or send as a
    different type; a missing field must degrade, never raise ``AttributeError``.
    """
    return dict(value) if isinstance(value, dict) else {}


def _content_text(content: object) -> str:
    """Flatten a content field that may be a string, ``None`` or a parts list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return ""


def _finish_reason(raw: object) -> FinishReason:
    if raw is None:
        return FinishReason.STOP
    return _FINISH_REASONS.get(str(raw), FinishReason.ERROR)


def _tool_calls(raw: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(_tool_call(entry) for entry in raw if isinstance(entry, dict))


def _tool_call(entry: Mapping[str, Any]) -> ToolCall:
    function = _as_object(entry.get("function"))
    arguments = function.get("arguments", "")
    return ToolCall(
        id=str(entry.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
    )


def _usage(raw: object) -> TokenUsage:
    if not isinstance(raw, Mapping):
        return TokenUsage()
    return TokenUsage(
        input_tokens=_non_negative_int(raw.get("prompt_tokens")),
        output_tokens=_non_negative_int(raw.get("completion_tokens")),
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _port_conformance(
    provider: OpenAICompatibleLLMProvider,
    factory: HttpLLMProviderFactory,
) -> tuple[LLMProvider, LLMProviderFactory]:
    """Static-only guard: mypy fails here if either class drifts from its port."""
    return provider, factory
