"""Readiness and smoke testing against the local vLLM server.

Readiness here means "this worker can serve a request", not "a process exists".
The distinction matters: vLLM accepts a TCP connection long before the weights
are resident, and a worker that reports ready too early gets handed a job it
drops.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import httpx

from worker.config import WorkerConfig

__all__ = [
    "NotReadyError",
    "ReadinessResult",
    "SmokeResult",
    "list_models",
    "smoke_test",
    "wait_until_ready",
]

# Deliberately tiny: the smoke test proves the pipeline works, not that the
# model is clever. Every extra token is GPU time spent on a health check.
SMOKE_PROMPT = "Write a valid C function that adds two integers."
SMOKE_MAX_TOKENS = 64


class NotReadyError(RuntimeError):
    """The server did not become able to serve within its deadline."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint

    def render(self) -> str:
        lines = [f"readiness check failed: {self}"]
        if self.hint:
            lines.append(f"  fix: {self.hint}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Outcome of polling the server."""

    ready: bool
    models: tuple[str, ...] = ()
    waited_seconds: float = 0.0
    detail: str | None = None

    def render(self) -> str:
        if self.ready:
            return f"ready after {self.waited_seconds:.0f}s, serving {', '.join(self.models)}"
        return f"not ready after {self.waited_seconds:.0f}s: {self.detail or 'unknown reason'}"


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Outcome of one real completion."""

    passed: bool
    model: str = ""
    output: str = ""
    finish_reason: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    detail: str | None = None

    def render(self) -> list[str]:
        if not self.passed:
            return [f"smoke test FAILED: {self.detail}"]
        preview = self.output.strip().replace("\n", " ")[:120]
        return [
            "smoke test passed",
            f"  model            {self.model}",
            f"  latency          {self.latency_seconds:.2f}s",
            f"  tokens           {self.prompt_tokens} in / {self.completion_tokens} out",
            f"  finish reason    {self.finish_reason}",
            f"  output (trimmed) {preview}",
        ]


def _headers(config: WorkerConfig) -> dict[str, str]:
    if config.vllm_api_key:
        return {"authorization": f"Bearer {config.vllm_api_key.reveal()}"}
    return {}


def list_models(config: WorkerConfig, *, timeout: float = 5.0) -> tuple[str, ...]:
    """Ask the server which models it serves.

    Raises on anything other than a well-formed 200: a 503 or an HTML error page
    from something else listening on the port must not read as success.
    """
    with httpx.Client(timeout=timeout, headers=_headers(config)) as client:
        response = client.get(f"{config.base_url}/v1/models")
    if response.status_code != httpx.codes.OK:
        raise NotReadyError(f"/v1/models returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise NotReadyError(f"/v1/models did not return JSON: {exc}") from exc
    data = payload.get("data")
    if not isinstance(data, list):
        raise NotReadyError("/v1/models returned no model list")
    return tuple(str(entry.get("id", "")) for entry in data if isinstance(entry, dict))


def wait_until_ready(
    config: WorkerConfig,
    *,
    timeout_seconds: float | None = None,
    poll_seconds: float | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda _message: None,
) -> ReadinessResult:
    """Poll until the server lists the model it was asked to serve.

    Checking the model *name* and not merely a 200 is what catches the case
    where vLLM came up serving something else entirely — a stale argument, a
    wrong snapshot path — which would otherwise fail much later as a puzzling
    404 from the orchestrator.
    """
    deadline_total = timeout_seconds or config.readiness_timeout_seconds
    interval = poll_seconds or config.readiness_poll_seconds
    started = now()
    deadline = started + deadline_total
    expected = config.public_model_name
    detail = "no attempt completed"

    while now() < deadline:
        try:
            models = list_models(config)
        except (NotReadyError, httpx.HTTPError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
        else:
            if expected in models:
                return ReadinessResult(True, models, now() - started)
            detail = f"serving {models or '()'}, expected {expected!r}"
        remaining = deadline - now()
        if remaining <= 0:
            break
        log(f"not ready yet ({detail}); {remaining:.0f}s left")
        sleep(min(interval, remaining))

    return ReadinessResult(False, (), now() - started, detail)


def smoke_test(
    config: WorkerConfig,
    *,
    timeout: float = 120.0,
    max_tokens: int = SMOKE_MAX_TOKENS,
    prompt: str = SMOKE_PROMPT,
    now: Callable[[], float] = time.monotonic,
) -> SmokeResult:
    """Issue one small completion and validate the whole response.

    Validates the shape as well as the status: a 200 carrying an empty string is
    a broken worker, and letting it pass would advertise capacity that produces
    nothing.
    """
    started = now()
    body = {
        "model": config.public_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        with httpx.Client(timeout=timeout, headers=_headers(config)) as client:
            response = client.post(f"{config.base_url}/v1/chat/completions", json=body)
    except httpx.HTTPError as exc:
        return SmokeResult(False, detail=f"request failed: {exc}", latency_seconds=now() - started)

    latency = now() - started
    if response.status_code != httpx.codes.OK:
        return SmokeResult(
            False,
            detail=f"HTTP {response.status_code}: {response.text[:300]}",
            latency_seconds=latency,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        return SmokeResult(False, detail=f"response was not JSON: {exc}", latency_seconds=latency)

    problems = _validate_completion(payload, expected_model=config.public_model_name)
    if problems:
        return SmokeResult(False, detail="; ".join(problems), latency_seconds=latency)

    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return SmokeResult(
        passed=True,
        model=str(payload.get("model", "")),
        output=str(choice["message"]["content"]),
        finish_reason=str(choice.get("finish_reason", "")),
        latency_seconds=latency,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
    )


def _validate_completion(payload: object, *, expected_model: str) -> list[str]:
    """Everything that must hold for the answer to count as a working worker."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["response was not a JSON object"]
    if payload.get("object") != "chat.completion":
        problems.append(f"unexpected object {payload.get('object')!r}")
    model = payload.get("model")
    if model != expected_model:
        problems.append(f"answered as {model!r}, expected {expected_model!r}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return [*problems, "response carried no choices"]
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return [*problems, "the first choice carried no message"]
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        problems.append("the model produced no output")
    return problems


@dataclass(frozen=True, slots=True)
class HealthSummary:
    """Liveness only: is something answering on the port at all."""

    alive: bool
    detail: str = ""
    models: Sequence[str] = field(default_factory=tuple)


def health(config: WorkerConfig, *, timeout: float = 3.0) -> HealthSummary:
    """Cheap liveness probe, suitable for a container HEALTHCHECK.

    Separate from readiness on purpose: during a long model load the process is
    alive and must not be restarted by an impatient supervisor, yet it is not
    ready to serve.
    """
    try:
        with httpx.Client(timeout=timeout, headers=_headers(config)) as client:
            response = client.get(f"{config.base_url}/health")
        if response.status_code == httpx.codes.OK:
            return HealthSummary(True, "healthy")
        return HealthSummary(False, f"/health returned HTTP {response.status_code}")
    except httpx.HTTPError as exc:
        return HealthSummary(False, f"{type(exc).__name__}: {exc}")
