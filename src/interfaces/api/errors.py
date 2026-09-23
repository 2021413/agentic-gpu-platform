"""RFC 9457 problem details, and the only place HTTP status codes are decided.

The domain raises errors that carry a stable ``code`` and no HTTP knowledge
(``src/domain/exceptions.py``). This module is the single translation table from
that code to a status. Keeping it in one dict — rather than spread over routes —
means a new domain error surfaces here, once, and that the same failure always
produces the same status whichever route triggered it.

Every response uses ``application/problem+json`` with the RFC 9457 members
(``type``, ``title``, ``status``, ``detail``, ``instance``) plus two extensions:

* ``code`` — the domain code, so a client branches on a stable token instead of
  parsing prose or overloading the status;
* ``details`` — the structured context the error carried (ids, states, limits).

``request_id`` is echoed as well: a support conversation about a 500 starts by
finding the matching access log line.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from domain.exceptions import DomainError
from interfaces.api.middleware.correlation import REQUEST_ID_SCOPE_KEY, current_request_id

__all__ = [
    "PROBLEM_CONTENT_TYPE",
    "STATUS_BY_CODE",
    "ApiError",
    "ServiceAuthError",
    "install_error_handlers",
    "problem_response",
    "status_for_code",
]

PROBLEM_CONTENT_TYPE: Final = "application/problem+json"

DEFAULT_PROBLEM_BASE_URI: Final = "/problems"
"""Relative URI reference, resolved by the client against the request URL.

RFC 9457 allows a relative ``type``; using one avoids baking a hostname the
deployment may not own into every error body.
"""

_RETRY_AFTER_SECONDS: Final = 5
"""Advertised on 503: the pool is elastic, a worker may join within seconds."""


class ApiError(Exception):
    """A protocol-level failure raised by the interfaces layer itself.

    Mirrors ``DomainError``'s shape (``code`` / ``message`` / ``details``) so a
    single renderer serves both. Use it only for HTTP concerns — authentication,
    malformed protocol usage — never to express a business rule.
    """

    def __init__(self, code: str, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details


class ServiceAuthError(ApiError):
    """The internal API refused a caller (spec section 19).

    Two distinct codes, because they mean different things to the caller: a
    missing credential is fixable by sending one, a wrong one is not.
    """

    @classmethod
    def missing(cls) -> ServiceAuthError:
        return cls("unauthenticated", "a service token is required")

    @classmethod
    def invalid(cls) -> ServiceAuthError:
        return cls("permission_denied", "the supplied service token is not accepted")


# ---------------------------------------------------------------------------
# code -> status
# ---------------------------------------------------------------------------
STATUS_BY_CODE: Final[Mapping[str, int]] = {
    # The resource genuinely does not exist. Nothing else can be said.
    "not_found": 404,
    # 409 Conflict: the request is well-formed and the resource exists, but its
    # current state forbids the transition. Retrying the same request unchanged
    # will keep failing until the resource moves — that is exactly "conflict".
    "invalid_state_transition": 409,
    "run_not_modifiable": 409,
    "project_not_modifiable": 409,
    "run_cancelled": 409,
    "job_lease_expired": 409,
    "job_not_retryable": 409,
    # A named worker exists but cannot take work: again a state conflict about a
    # specific resource, not a capacity problem of the platform.
    "worker_unavailable": 409,
    # Same key, different body: the caller contradicted itself (spec section 35).
    "idempotency_conflict": 409,
    # 422: the syntax was valid, the *content* is not usable. Both of these are
    # semantic failures of a payload the platform accepted at parse time.
    "plan_invalid": 422,
    "structured_output_invalid": 422,
    # 504: the platform depends on an upstream inference server that did not
    # answer in time. A gateway timeout says "upstream, not you" and is
    # idempotent-retry friendly.
    "llm_timeout": 504,
    # 502: the inference engine answered, badly, or could not be reached. The
    # control plane is healthy; its upstream is not.
    "inference_failed": 502,
    # 503: no compatible worker *right now*. The fleet is elastic, so this is
    # explicitly temporary and carries Retry-After.
    "no_compatible_worker": 503,
    # Server-side machinery that failed on our side of the boundary.
    "tool_execution_failed": 500,
    "workspace_error": 500,
    "candidate_error": 500,
    "domain_error": 500,
    # Interfaces-layer codes.
    "unauthenticated": 401,
    "permission_denied": 403,
    "validation_error": 422,
    "not_ready": 503,
    "internal_error": 500,
}

_TITLES: Final[Mapping[str, str]] = {
    "not_found": "Resource not found",
    "invalid_state_transition": "Invalid state transition",
    "run_not_modifiable": "Run is no longer modifiable",
    "project_not_modifiable": "Project is in use by a run",
    "run_cancelled": "Run has been cancelled",
    "job_lease_expired": "Job lease expired",
    "job_not_retryable": "Job cannot be retried",
    "worker_unavailable": "Worker unavailable",
    "idempotency_conflict": "Idempotency key conflict",
    "plan_invalid": "Plan is not executable",
    "structured_output_invalid": "Model output failed validation",
    "llm_timeout": "Inference timed out",
    "inference_failed": "Inference backend failed",
    "no_compatible_worker": "No compatible worker available",
    "tool_execution_failed": "Tool execution failed",
    "workspace_error": "Workspace failure",
    "candidate_error": "Candidate failure",
    "domain_error": "Internal error",
    "unauthenticated": "Service authentication required",
    "permission_denied": "Service token rejected",
    "validation_error": "Request validation failed",
    "not_ready": "Service not ready",
    "internal_error": "Internal error",
}

_FALLBACK_STATUS: Final = 500

_logger = logging.getLogger("interfaces.api.errors")


def status_for_code(code: str) -> int:
    """Status for a domain code; unknown codes are server faults, not 400s.

    An error the API has never heard of means the platform raised something the
    boundary was not updated for. That is a defect on our side, so it must be a
    5xx: silently answering 400 would blame the client for our omission.
    """
    return STATUS_BY_CODE.get(code, _FALLBACK_STATUS)


def problem_response(
    *,
    status: int,
    code: str,
    detail: str,
    instance: str,
    details: Mapping[str, Any] | None = None,
    title: str | None = None,
    problem_base_uri: str = DEFAULT_PROBLEM_BASE_URI,
    headers: Mapping[str, str] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Render one RFC 9457 document."""
    body: dict[str, Any] = {
        "type": f"{problem_base_uri.rstrip('/')}/{code}",
        "title": title or _TITLES.get(code, "Error"),
        "status": status,
        "detail": detail,
        "instance": instance,
        "code": code,
    }
    if details:
        body["details"] = dict(details)
    correlation = request_id or current_request_id()
    if correlation:
        body["request_id"] = correlation

    response_headers = dict(headers or {})
    if status == STATUS_BY_CODE["unauthenticated"]:
        response_headers.setdefault("WWW-Authenticate", 'Bearer realm="internal"')
    if status == STATUS_BY_CODE["no_compatible_worker"]:
        response_headers.setdefault("Retry-After", str(_RETRY_AFTER_SECONDS))

    return JSONResponse(
        status_code=status,
        content=body,
        media_type=PROBLEM_CONTENT_TYPE,
        headers=response_headers,
    )


_Handler = Callable[[Request, Exception], Response | Awaitable[Response]]


def _correlation_of(request: Request) -> str:
    """The request id, read from the scope first.

    An unhandled exception is rendered by Starlette's ``ServerErrorMiddleware``,
    which sits outside the correlation middleware: by then the context variable
    is gone but the scope still carries the id.
    """
    stored = request.scope.get(REQUEST_ID_SCOPE_KEY)
    return str(stored) if stored else current_request_id()


def install_error_handlers(
    app: FastAPI, *, problem_base_uri: str = DEFAULT_PROBLEM_BASE_URI
) -> None:
    """Register every handler that turns an exception into problem+json.

    Handlers are closures over ``problem_base_uri`` so the ``type`` URI is a
    deployment choice rather than a constant frozen into the code.
    """

    async def handle_domain_error(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, DomainError):  # pragma: no cover - defensive
            raise exc
        status = status_for_code(exc.code)
        if status >= _FALLBACK_STATUS:
            # A 5xx means the platform, not the caller, is at fault: it must
            # leave a stack trace behind, unlike an expected 404 or 409.
            _logger.exception("unmapped_or_server_side_domain_error", exc_info=exc)
        return problem_response(
            status=status,
            code=exc.code,
            detail=exc.message,
            instance=request.url.path,
            details=exc.details,
            problem_base_uri=problem_base_uri,
            request_id=_correlation_of(request),
        )

    async def handle_api_error(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, ApiError):  # pragma: no cover - defensive
            raise exc
        return problem_response(
            status=status_for_code(exc.code),
            code=exc.code,
            detail=exc.message,
            instance=request.url.path,
            details=exc.details,
            problem_base_uri=problem_base_uri,
            request_id=_correlation_of(request),
        )

    async def handle_validation_error(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, RequestValidationError):  # pragma: no cover - defensive
            raise exc
        return problem_response(
            status=STATUS_BY_CODE["validation_error"],
            code="validation_error",
            detail="the request payload failed validation",
            instance=request.url.path,
            # ``jsonable_encoder``-free: pydantic v2 errors may carry exception
            # objects in ``ctx``, which are not JSON-serializable.
            details={"errors": [_render_validation_error(e) for e in exc.errors()]},
            problem_base_uri=problem_base_uri,
            request_id=_correlation_of(request),
        )

    async def handle_http_exception(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, HTTPException):  # pragma: no cover - defensive
            raise exc
        # Raised by Starlette itself (404 on an unknown path, 405 on a wrong
        # method). Rendering them as problem+json too keeps clients on a single
        # error format, whatever produced the failure.
        return problem_response(
            status=exc.status_code,
            code=_code_for_status(exc.status_code),
            detail=str(exc.detail),
            instance=request.url.path,
            problem_base_uri=problem_base_uri,
            headers=exc.headers,
            request_id=_correlation_of(request),
        )

    async def handle_unexpected_error(request: Request, exc: Exception) -> Response:
        _logger.exception("unhandled_api_error", exc_info=exc)
        return problem_response(
            status=STATUS_BY_CODE["internal_error"],
            code="internal_error",
            # Deliberately opaque: an internal message may quote a prompt, a
            # path or a connection string. The request id is the bridge to the
            # logs where the real cause lives.
            detail="the request could not be completed",
            instance=request.url.path,
            problem_base_uri=problem_base_uri,
            request_id=_correlation_of(request),
        )

    handlers: list[tuple[type[Exception], _Handler]] = [
        (DomainError, handle_domain_error),
        (ApiError, handle_api_error),
        (RequestValidationError, handle_validation_error),
        (HTTPException, handle_http_exception),
        (Exception, handle_unexpected_error),
    ]
    for exc_type, handler in handlers:
        app.add_exception_handler(exc_type, handler)


_STATUS_CODES: Final[Mapping[int, str]] = {
    401: "unauthenticated",
    403: "permission_denied",
    404: "not_found",
    422: "validation_error",
    503: "not_ready",
}


def _code_for_status(status: int) -> str:
    """Give a Starlette-raised status a code from the same vocabulary."""
    return _STATUS_CODES.get(status, "internal_error" if status >= _FALLBACK_STATUS else "error")


def _render_validation_error(error: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the JSON-safe parts of a pydantic error.

    ``input`` is echoed back as a string: it is untrusted and may be any object,
    and reflecting it verbatim into a response is how error bodies turn into
    injection vectors.
    """
    return {
        "location": [str(part) for part in error.get("loc", ())],
        "message": str(error.get("msg", "")),
        "type": str(error.get("type", "")),
    }
