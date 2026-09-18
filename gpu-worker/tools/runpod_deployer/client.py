"""HTTP access to RunPod, and to the worker once RunPod has placed it.

Two very different servers are reached from here and they are kept apart on
purpose:

* the RunPod control plane -- ``https://rest.runpod.io/v1`` for Pods and network
  volumes (Bearer token), ``https://api.runpod.io/graphql`` for the GPU type
  catalogue, which the REST API does not expose;
* the worker itself, over whichever public URL RunPod ended up giving us.

The control-plane credential is read from ``RUNPOD_API_KEY`` and from nowhere
else: no flag, no file, no default. A flag would put it in shell history and in
``ps``; a file would make it something to forget in a repository. Everything
that leaves this module -- exception text, logs, ``repr`` -- goes through
:func:`redact` first, because RunPod's own error bodies happily echo back what
you sent them.

Verified against https://rest.runpod.io/v1/openapi.json (Runpod API 0.1.0) and
https://graphql-spec.runpod.io/ on 2026-09-18.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final

import httpx

from .models import (
    PROXY_TIMEOUT_SECONDS,
    GpuType,
    NetworkVolume,
    PodSpec,
    PodState,
    SecretValue,
)

__all__ = [
    "ApiError",
    "AuthenticationError",
    "GpuUnavailableError",
    "MissingApiKeyError",
    "NotReadyError",
    "QuotaError",
    "ReadinessReport",
    "RunPodClient",
    "RunPodError",
    "ServiceError",
    "SmokeReport",
    "TransportError",
    "VolumeNotFoundError",
    "api_key_from_environment",
    "list_models",
    "measure_first_byte",
    "redact",
    "smoke_test",
    "wait_until_ready",
]

API_KEY_ENV: Final = "RUNPOD_API_KEY"
REST_BASE_URL: Final = "https://rest.runpod.io/v1"
GRAPHQL_URL: Final = "https://api.runpod.io/graphql"

# Long enough to survive a slow control plane, short enough that a wedged
# connection does not hold a deployment hostage. A GPU is billed meanwhile.
DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)

RETRYABLE_STATUS: Final = frozenset(
    {
        httpx.codes.REQUEST_TIMEOUT,
        httpx.codes.TOO_EARLY,
        httpx.codes.TOO_MANY_REQUESTS,
        httpx.codes.INTERNAL_SERVER_ERROR,
        httpx.codes.BAD_GATEWAY,
        httpx.codes.SERVICE_UNAVAILABLE,
        httpx.codes.GATEWAY_TIMEOUT,
    }
)
# 401 and 403 are deliberately absent above, and asserted absent below: a
# rejected key is rejected again three times later, and hammering an auth
# endpoint is how an account gets rate limited on top of being broken.
NEVER_RETRIED: Final = frozenset({httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN})

MAX_ERROR_CHARS: Final = 400

# Cloudflare's own status for "the origin took longer than 100 seconds". It is
# not an HTTP standard code, so httpx.codes does not name it.
CLOUDFLARE_TIMEOUT_STATUS: Final = 524

SMOKE_PROMPT: Final = "Write a valid C function that adds two integers."
SMOKE_MAX_TOKENS: Final = 64

MIN_REDACTABLE_LENGTH: Final = 4

_REDACTED: Final = "***"
# RunPod keys are `rpa_` followed by base32-ish characters; masked even when the
# client does not know the value, e.g. a key pasted into a Pod's own logs.
_KEY_PATTERN: Final = re.compile(r"\brpa_[A-Za-z0-9]{8,}\b")
_QUERY_PATTERN: Final = re.compile(r"((?:api_key|apikey|token|key)=)([^&\s\"']+)", re.IGNORECASE)
_BEARER_PATTERN: Final = re.compile(r"(Bearer\s+)(\S+)", re.IGNORECASE)


def redact(text: str, *, secrets: Iterable[str] = ()) -> str:
    """Remove credentials from a string that is about to be shown to someone.

    Four things are masked: the exact secrets we hold, anything shaped like a
    RunPod key, anything in a ``key=`` query parameter (the GraphQL endpoint
    takes the credential that way), and any ``Bearer`` header echoed back. The
    first covers what we sent, the rest cover what someone else may have put in
    a response body we are about to re-raise.
    """
    cleaned = text
    for secret in secrets:
        if secret and len(secret) >= MIN_REDACTABLE_LENGTH:
            cleaned = cleaned.replace(secret, _REDACTED)
    cleaned = _KEY_PATTERN.sub(_REDACTED, cleaned)
    cleaned = _QUERY_PATTERN.sub(lambda m: f"{m.group(1)}{_REDACTED}", cleaned)
    return _BEARER_PATTERN.sub(lambda m: f"{m.group(1)}{_REDACTED}", cleaned)


# ----------------------------------------------------------------------
# errors: each one says what an operator should do next
# ----------------------------------------------------------------------
class RunPodError(RuntimeError):
    """Base class. The message is already redacted by construction."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def render(self) -> str:
        lines = [f"{type(self).__name__}: {self}"]
        if self.hint:
            lines.append(f"  fix: {self.hint}")
        return "\n".join(lines)


class MissingApiKeyError(RunPodError):
    """No credential in the environment."""


class AuthenticationError(RunPodError):
    """The credential exists and RunPod refused it. Never retried."""


class QuotaError(RunPodError):
    """Out of credit, or over a limit. Retrying costs time, not success."""


class GpuUnavailableError(RunPodError):
    """No machine of the requested type, in the requested place, right now."""


class VolumeNotFoundError(RunPodError):
    """The network volume does not exist, or is not in reach of this account."""


class PodNotFoundError(RunPodError):
    """No such Pod. Also what a terminated Pod eventually looks like."""


class ApiError(RunPodError):
    """A refusal we could not classify. Carries the status and the body."""

    def __init__(self, message: str, *, status_code: int, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.status_code = status_code


class ServiceError(RunPodError):
    """RunPod is broken or overloaded, after the retries were exhausted."""


class TransportError(RunPodError):
    """The request never got an answer: DNS, TLS, connection, timeout."""


class NotReadyError(RunPodError):
    """The worker did not become able to serve within its deadline.

    Mirrors ``worker.readiness.NotReadyError``; kept separate so the deployer
    does not need ``src/`` importable on an operator's laptop.
    """


def api_key_from_environment(environ: Mapping[str, str]) -> str:
    """The one way a credential enters this program."""
    value = (environ.get(API_KEY_ENV) or "").strip()
    if not value:
        raise MissingApiKeyError(
            f"{API_KEY_ENV} is not set",
            hint=(
                f"export {API_KEY_ENV}=... in this shell "
                "(https://console.runpod.io/user/settings -> API Keys). "
                "The deployer never accepts the key as an argument or from a file."
            ),
        )
    return value


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded, explicit, and never applied to an authentication failure."""

    attempts: int = 3
    backoff_seconds: float = 2.0
    backoff_factor: float = 2.0

    def delay(self, attempt: int) -> float:
        return self.backoff_seconds * (self.backoff_factor ** (attempt - 1))


class RunPodClient:
    """The RunPod control plane, as the deployer needs it.

    The credential is read from the environment inside the constructor, so that
    there is no parameter anyone could be tempted to pass one through. Tests
    inject a fake ``environ`` mapping, which is the same rule applied to a
    different environment, not an exception to it.
    """

    def __init__(
        self,
        *,
        environ: Mapping[str, str],
        base_url: str = REST_BASE_URL,
        graphql_url: str = GRAPHQL_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self._key = SecretValue(api_key_from_environment(environ))
        self._base_url = base_url.rstrip("/")
        self._graphql_url = graphql_url
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._log = log
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._key.reveal()}",
                "Content-Type": "application/json",
                "User-Agent": "gpu-worker-runpod-deployer/1.0",
            },
        )

    # -- lifecycle -------------------------------------------------------
    def __repr__(self) -> str:
        return f"RunPodClient(base_url={self._base_url!r}, api_key=<redacted>)"

    __str__ = __repr__

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RunPodClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- operations ------------------------------------------------------
    def create_pod(self, spec: PodSpec) -> PodState:
        """``POST /pods``. Returns the Pod as created, usually without an IP yet."""
        body = spec.to_create_body()
        payload = self._request("POST", "/pods", json_body=body, context="create pod")
        if not isinstance(payload, dict) or not payload.get("id"):
            raise ApiError(
                "RunPod accepted the Pod but returned no id",
                status_code=int(httpx.codes.CREATED),
                hint="check https://console.runpod.io/pods before creating another one",
            )
        return PodState.from_api(payload)

    def get_pod(self, pod_id: str) -> PodState:
        """``GET /pods/{podId}`` with the machine block, for the GPU name."""
        payload = self._request(
            "GET",
            f"/pods/{pod_id}",
            params={"includeMachine": "true", "includeNetworkVolume": "true"},
            context=f"read pod {pod_id}",
        )
        if not isinstance(payload, dict):
            raise ApiError(f"unexpected response for pod {pod_id}", status_code=int(httpx.codes.OK))
        return PodState.from_api(payload)

    def terminate_pod(self, pod_id: str) -> None:
        """``DELETE /pods/{podId}``. Irreversible; the caller owns the confirmation."""
        self._request("DELETE", f"/pods/{pod_id}", context=f"terminate pod {pod_id}")

    def get_network_volume(self, volume_id: str) -> NetworkVolume:
        """``GET /networkvolumes/{id}``, used to fail before renting a GPU.

        A volume lives in exactly one data center and a Pod that wants it must
        be placed there, so its ``dataCenterId`` is also what the deployer uses
        to constrain placement.
        """
        payload = self._request(
            "GET",
            f"/networkvolumes/{volume_id}",
            context=f"read network volume {volume_id}",
            resource="volume",
        )
        if not isinstance(payload, dict):
            raise VolumeNotFoundError(f"network volume {volume_id} returned no description")
        return NetworkVolume.from_api(payload)

    def list_gpu_types(self) -> tuple[GpuType, ...]:
        """The GPU catalogue, from GraphQL.

        The REST API has no GPU-type endpoint (no such path in
        https://rest.runpod.io/v1/openapi.json); ``gpuTypeId`` is only an enum
        in its schema, with no availability or price. The GraphQL ``gpuTypes``
        query is the documented way to get the live list.
        """
        query = (
            "query GpuTypes { gpuTypes { id displayName memoryInGb "
            "secureCloud communityCloud securePrice communityPrice } }"
        )
        payload = self._graphql(query)
        types = payload.get("gpuTypes")
        if not isinstance(types, list):
            raise ApiError(
                "the GraphQL API returned no gpuTypes list", status_code=int(httpx.codes.OK)
            )
        return tuple(GpuType.from_api(entry) for entry in types if isinstance(entry, dict))

    # -- transport -------------------------------------------------------
    def _graphql(self, query: str) -> dict[str, Any]:
        # The GraphQL endpoint is documented as taking the credential in the
        # query string (docs.runpod.io/sdks/graphql/manage-pods). That is one
        # more place it can leak, which is exactly why every string leaving
        # this class is passed through redact().
        response = self._send(
            "POST",
            self._graphql_url,
            json_body={"query": query},
            params={"api_key": self._key.reveal()},
            context="list gpu types",
        )
        payload = self._decode(response, context="list gpu types")
        if not isinstance(payload, dict):
            raise ApiError("the GraphQL API returned no object", status_code=response.status_code)
        errors = payload.get("errors")
        if errors:
            message = self._clean(json.dumps(errors))
            if "unauthor" in message.lower() or "invalid" in message.lower():
                raise AuthenticationError(
                    f"GraphQL refused the credential: {message}",
                    hint=f"check {API_KEY_ENV}; a REST key and a GraphQL key are the same key",
                )
            raise ApiError(f"GraphQL error: {message}", status_code=response.status_code)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ApiError("the GraphQL API returned no data", status_code=response.status_code)
        return data

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        context: str,
        resource: str = "pod",
    ) -> Any:
        response = self._send(
            method,
            f"{self._base_url}{path}",
            json_body=json_body,
            params=params,
            context=context,
            resource=resource,
        )
        if response.status_code == httpx.codes.NO_CONTENT or not response.content:
            return None
        return self._decode(response, context=context)

    def _send(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Any] | None,
        params: Mapping[str, str] | None,
        context: str,
        resource: str = "pod",
    ) -> httpx.Response:
        """One call, with bounded retries on transient failures only."""
        last_error: RunPodError | None = None
        for attempt in range(1, self._retry.attempts + 1):
            try:
                response = self._client.request(method, url, json=json_body, params=params)
            except httpx.HTTPError as exc:
                # An httpx message can carry the full URL, credential included.
                last_error = TransportError(
                    f"{context}: {type(exc).__name__}: {self._clean(str(exc))}",
                    hint="network or RunPod outage; the Pod may still have been created",
                )
            else:
                if response.status_code in NEVER_RETRIED:
                    raise self._translate(response, context=context, resource=resource)
                if response.status_code not in RETRYABLE_STATUS:
                    if response.is_success:
                        return response
                    raise self._translate(response, context=context, resource=resource)
                last_error = self._translate(response, context=context, resource=resource)

            if attempt < self._retry.attempts:
                delay = self._retry.delay(attempt)
                self._log(f"{context}: {last_error}; retrying in {delay:.0f}s")
                self._sleep(delay)

        assert last_error is not None  # noqa: S101 - the loop always sets it
        raise last_error

    def _decode(self, response: httpx.Response, *, context: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                f"{context}: response was not JSON ({self._clean(str(exc))})",
                status_code=response.status_code,
            ) from exc

    def _clean(self, text: str) -> str:
        return redact(text, secrets=(self._key.reveal(),))[:MAX_ERROR_CHARS]

    def _translate(self, response: httpx.Response, *, context: str, resource: str) -> RunPodError:
        """Turn a refusal into something an operator can act on."""
        detail = self._clean(_body_message(response))
        status = response.status_code
        prefix = f"{context} failed with HTTP {status}"

        if status in NEVER_RETRIED:
            return AuthenticationError(
                f"{prefix}: RunPod rejected the credential. {detail}",
                hint=(
                    f"{API_KEY_ENV} is set but invalid, revoked, or lacks write access. "
                    "Re-issue it at https://console.runpod.io/user/settings."
                ),
            )
        if status == httpx.codes.BAD_REQUEST:
            return _refused_request(prefix, detail)
        if status == httpx.codes.NOT_FOUND:
            return _missing_resource(prefix, detail, resource)
        if status in (httpx.codes.PAYMENT_REQUIRED, httpx.codes.TOO_MANY_REQUESTS):
            rate_limited = status == httpx.codes.TOO_MANY_REQUESTS
            return QuotaError(
                f"{prefix}: {'rate limited' if rate_limited else 'refused for billing reasons'}."
                f" {detail}",
                hint=(
                    "slow the polling down; this call was already retried"
                    if rate_limited
                    else "top up credit, or lower gpuCount / containerDiskInGb"
                ),
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            return ServiceError(
                f"{prefix}: RunPod is failing. {detail}",
                hint="check https://uptime.runpod.io before retrying by hand",
            )
        return ApiError(f"{prefix}: {detail}", status_code=status)


def _refused_request(prefix: str, detail: str) -> RunPodError:
    """A 400 is where RunPod says what it could not give us, in prose."""
    lowered = detail.lower()
    if "volume" in lowered:
        return VolumeNotFoundError(
            f"{prefix}: the network volume was refused. {detail}",
            hint=(
                "a volume only exists in one data center: the Pod must be created there, "
                "and that data center must have the GPU type free"
            ),
        )
    if any(word in lowered for word in ("gpu", "instance", "capacity", "availab")):
        return GpuUnavailableError(
            f"{prefix}: no machine matches the request. {detail}",
            hint=(
                "run `gpu-types` to see what is free, widen --gpu-type, or drop "
                "--data-center if no volume forces one"
            ),
        )
    return ApiError(
        f"{prefix}: {detail}",
        status_code=int(httpx.codes.BAD_REQUEST),
        hint="check the request body",
    )


def _missing_resource(prefix: str, detail: str, resource: str) -> RunPodError:
    if resource == "volume":
        return VolumeNotFoundError(
            f"{prefix}: no such network volume. {detail}",
            hint="check the id at https://console.runpod.io/user/storage",
        )
    return PodNotFoundError(
        f"{prefix}: no such Pod. {detail}",
        hint="it may already be terminated; ids are not reused",
    )


def _body_message(response: httpx.Response) -> str:
    """The most useful sentence in an error body, whatever shape it has."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:MAX_ERROR_CHARS]
    if isinstance(payload, dict):
        for key in ("error", "message", "detail", "errors"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
            if value:
                return json.dumps(value)
    return json.dumps(payload)[:MAX_ERROR_CHARS]


# ----------------------------------------------------------------------
# the worker side: the same validation as src/worker/readiness.py, aimed at
# the public URL instead of localhost
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Outcome of polling the public endpoint until the model is listed."""

    ready: bool
    models: tuple[str, ...] = ()
    waited_seconds: float = 0.0
    detail: str | None = None

    def render(self) -> str:
        if self.ready:
            return f"ready after {self.waited_seconds:.0f}s, serving {', '.join(self.models)}"
        return f"not ready after {self.waited_seconds:.0f}s: {self.detail or 'unknown reason'}"


@dataclass(frozen=True, slots=True)
class SmokeReport:
    """Outcome of one real completion through the public URL."""

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


@dataclass(frozen=True, slots=True)
class FirstByteReport:
    """How long the first streamed byte took, and through which route.

    This is the number that decides whether the 100 s proxy ceiling is a real
    problem: if the first token arrives in a couple of seconds, a streaming
    client survives a long generation behind Cloudflare; if it does not, the
    proxy is unusable for anything but short answers.
    """

    ok: bool
    seconds: float = 0.0
    total_seconds: float = 0.0
    detail: str | None = None


def _auth_headers(api_key: SecretValue | None) -> dict[str, str]:
    if api_key and api_key.reveal():
        return {"Authorization": f"Bearer {api_key.reveal()}"}
    return {}


def list_models(
    base_url: str,
    *,
    api_key: SecretValue | None = None,
    timeout: float = 10.0,
) -> tuple[str, ...]:
    """``GET {base_url}/v1/models``, validated like the in-container check.

    Anything but a well-formed 200 raises: a 524 from the proxy, an HTML error
    page or a 503 from a half-started vLLM must not read as success.
    """
    with httpx.Client(timeout=timeout, headers=_auth_headers(api_key)) as client:
        response = client.get(f"{base_url.rstrip('/')}/v1/models")
    if response.status_code != httpx.codes.OK:
        raise NotReadyError(f"/v1/models returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise NotReadyError(f"/v1/models did not return JSON: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise NotReadyError("/v1/models returned no model list")
    return tuple(str(entry.get("id", "")) for entry in data if isinstance(entry, dict))


def wait_until_ready(
    base_url: str,
    model: str,
    *,
    timeout_seconds: float = 1800.0,
    poll_seconds: float = 10.0,
    api_key: SecretValue | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda _message: None,
) -> ReadinessReport:
    """Poll the public URL until it lists the model we asked for.

    Checking the name, not merely a 200, is what catches a Pod that came up
    serving something else -- a stale template, a wrong snapshot -- which would
    otherwise surface as a puzzling 404 from the orchestrator much later.
    """
    started = now()
    deadline = started + timeout_seconds
    detail = "no attempt completed"
    while now() < deadline:
        try:
            models = list_models(base_url, api_key=api_key)
        except (NotReadyError, httpx.HTTPError) as exc:
            detail = f"{type(exc).__name__}: {redact(str(exc))}"
        else:
            if model in models:
                return ReadinessReport(True, models, now() - started)
            detail = f"serving {models or '()'}, expected {model!r}"
        remaining = deadline - now()
        if remaining <= 0:
            break
        log(f"not ready yet ({detail}); {remaining:.0f}s left")
        sleep(min(poll_seconds, remaining))
    return ReadinessReport(False, (), now() - started, detail)


def smoke_test(
    base_url: str,
    model: str,
    *,
    api_key: SecretValue | None = None,
    timeout: float = 120.0,
    max_tokens: int = SMOKE_MAX_TOKENS,
    prompt: str = SMOKE_PROMPT,
    now: Callable[[], float] = time.monotonic,
) -> SmokeReport:
    """One small completion, validated in full.

    A 200 carrying an empty string is a broken worker; letting it pass would
    advertise capacity that produces nothing.
    """
    started = now()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        with httpx.Client(timeout=timeout, headers=_auth_headers(api_key)) as client:
            response = client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body)
    except httpx.HTTPError as exc:
        return SmokeReport(
            False,
            detail=f"request failed: {redact(str(exc))}",
            latency_seconds=now() - started,
        )

    latency = now() - started
    if response.status_code != httpx.codes.OK:
        return SmokeReport(
            False,
            detail=_status_detail(response),
            latency_seconds=latency,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        return SmokeReport(False, detail=f"response was not JSON: {exc}", latency_seconds=latency)

    problems = _validate_completion(payload, expected_model=model)
    if problems:
        return SmokeReport(False, detail="; ".join(problems), latency_seconds=latency)

    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return SmokeReport(
        passed=True,
        model=str(payload.get("model", "")),
        output=str(choice["message"]["content"]),
        finish_reason=str(choice.get("finish_reason", "")),
        latency_seconds=latency,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
    )


def measure_first_byte(
    base_url: str,
    model: str,
    *,
    api_key: SecretValue | None = None,
    timeout: float = 120.0,
    max_tokens: int = 32,
    prompt: str = SMOKE_PROMPT,
    now: Callable[[], float] = time.monotonic,
) -> FirstByteReport:
    """Stream a completion and time the first chunk that carries content."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    started = now()
    first: float | None = None
    try:
        with (
            httpx.Client(timeout=timeout, headers=_auth_headers(api_key)) as client,
            client.stream(
                "POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=body
            ) as response,
        ):
            if response.status_code != httpx.codes.OK:
                response.read()
                return FirstByteReport(False, detail=_status_detail(response))
            for chunk in response.iter_bytes():
                if chunk and first is None:
                    first = now() - started
                    # keep draining: the total tells us whether the whole
                    # generation would have fitted inside the proxy's ceiling.
    except httpx.HTTPError as exc:
        return FirstByteReport(
            False,
            seconds=first or 0.0,
            total_seconds=now() - started,
            detail=f"stream failed: {redact(str(exc))}",
        )
    if first is None:
        return FirstByteReport(
            False, total_seconds=now() - started, detail="stream carried no data"
        )
    return FirstByteReport(True, seconds=first, total_seconds=now() - started)


def _status_detail(response: httpx.Response) -> str:
    detail = f"HTTP {response.status_code}: {redact(response.text)[:300]}"
    if response.status_code in (CLOUDFLARE_TIMEOUT_STATUS, httpx.codes.GATEWAY_TIMEOUT):
        detail += (
            f" -- this is the Cloudflare proxy giving up after {PROXY_TIMEOUT_SECONDS}s, not the "
            "worker failing; use the direct TCP port for requests that take longer"
        )
    return detail


def _validate_completion(payload: object, *, expected_model: str) -> list[str]:
    """Everything that must hold for an answer to count as a working worker."""
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


@dataclass(slots=True)
class PodWaitReport:
    """How a Pod went from created to addressable."""

    state: PodState
    waited_seconds: float = 0.0
    detail: str | None = None
    reached: bool = False
    observations: list[str] = field(default_factory=list)


def wait_for_pod(
    client: RunPodClient,
    pod_id: str,
    *,
    timeout_seconds: float = 600.0,
    poll_seconds: float = 5.0,
    require_network: bool = True,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda _message: None,
) -> PodWaitReport:
    """Wait until RunPod reports the Pod running, and addressable if asked.

    ``desiredStatus == RUNNING`` alone is not enough when a direct TCP port is
    wanted: the port mapping and the public IP appear later, and an orchestrator
    handed a URL built from an empty mapping would get a connection refused.
    """
    started = now()
    deadline = started + timeout_seconds
    while True:
        state = client.get_pod(pod_id)
        if state.is_terminated:
            return PodWaitReport(state, now() - started, "the Pod is TERMINATED", reached=False)
        network_ok = state.has_network or not require_network
        if state.is_running and network_ok:
            return PodWaitReport(state, now() - started, None, reached=True)
        detail = (
            f"status {state.desired_status or 'unknown'}, "
            f"ip {state.public_ip or 'pending'}, mappings {state.port_mappings or '{}'}"
        )
        remaining = deadline - now()
        if remaining <= 0:
            return PodWaitReport(state, now() - started, detail, reached=False)
        log(f"pod not up yet ({detail}); {remaining:.0f}s left")
        sleep(min(poll_seconds, remaining))
