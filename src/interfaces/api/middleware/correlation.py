"""Request correlation, access logging and latency measurement.

One identifier follows a request through the API, the orchestrator logs and the
problem document returned on failure. Without it, a user report ("my run 500ed
at 14:32") cannot be tied to anything in a horizontally scaled deployment.

This is a *pure ASGI* middleware rather than a ``BaseHTTPMiddleware``: the
latter buffers the response through an anyio stream, which breaks long-lived
SSE responses and their disconnect detection. Everything here works at the
message level and never touches the body.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Final
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = [
    "REQUEST_ID_HEADER",
    "REQUEST_ID_SCOPE_KEY",
    "RequestContextMiddleware",
    "current_request_id",
]

REQUEST_ID_HEADER: Final = "X-Request-ID"
LATENCY_HEADER: Final = "X-Response-Time-Ms"

REQUEST_ID_SCOPE_KEY: Final = "interfaces.request_id"
"""Where the id is stored for code that cannot see this middleware's context.

Starlette's ``ServerErrorMiddleware`` wraps the *outside* of every user
middleware, so the handler for an unhandled exception runs after this one's
context has been torn down. The scope, on the other hand, is the same dict all
the way through, which makes it the only reliable carrier there.
"""

_MAX_REQUEST_ID_LENGTH: Final = 128

_request_id: ContextVar[str] = ContextVar("request_id", default="")

_access_logger = logging.getLogger("interfaces.api.access")


def current_request_id() -> str:
    """The identifier of the request being served, or ``""`` outside a request.

    Read by the error handlers so a problem document can name the exact log
    line that explains it.
    """
    return _request_id.get()


def _sanitize(candidate: str | None) -> str | None:
    """Accept a client-supplied correlation id only if it is safe to echo back.

    The header is untrusted input that ends up in log lines and in a response
    header: control characters would allow log forging and header injection, so
    anything outside printable ASCII, or anything absurdly long, is discarded
    rather than rejected — a bad correlation id must not fail a valid request.
    """
    if candidate is None:
        return None
    trimmed = candidate.strip()
    if not trimmed or len(trimmed) > _MAX_REQUEST_ID_LENGTH:
        return None
    if not all(0x20 <= ord(char) < 0x7F for char in trimmed):
        return None
    return trimmed


class RequestContextMiddleware:
    """Assign a request id, echo it back, log the access line with its latency."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        header_name: str = REQUEST_ID_HEADER,
        logger: logging.Logger | None = None,
    ) -> None:
        self._app = app
        self._header_name = header_name
        self._logger = logger or _access_logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = _sanitize(headers.get(self._header_name)) or uuid4().hex
        scope[REQUEST_ID_SCOPE_KEY] = request_id
        token = _request_id.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                elapsed_ms = (time.perf_counter() - started) * 1000
                mutable = MutableHeaders(scope=message)
                mutable[self._header_name] = request_id
                mutable[LATENCY_HEADER] = f"{elapsed_ms:.1f}"
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            # Logged in ``finally`` so a crashed or cancelled request still
            # leaves a trace; the duration then covers the whole body, which is
            # what matters for a streaming response.
            self._logger.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "query": scope.get("query_string", b"").decode("latin-1"),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "client": _client_of(scope),
                },
            )
            _request_id.reset(token)


def _client_of(scope: Scope) -> str:
    client = scope.get("client")
    if not client:
        return ""
    return str(client[0])
