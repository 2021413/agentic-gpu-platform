"""Structured logging (spec section 32).

Every line carries the identifiers that let a run be reconstructed after the
fact: request, run, project, job, worker, candidate. They travel in a
``contextvars`` context so a log statement deep inside a use case does not have
to thread them through every signature.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

__all__ = ["JsonFormatter", "bind_context", "configure_logging", "current_context"]

# ``None`` rather than ``{}``: a mutable default on a ContextVar is shared by
# every context that never set one.
_CORRELATION: ContextVar[Mapping[str, str] | None] = ContextVar("correlation", default=None)
_EMPTY: Mapping[str, str] = MappingProxyType({})

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


def current_context() -> Mapping[str, str]:
    return _CORRELATION.get() or _EMPTY


@contextmanager
def bind_context(**values: str) -> Iterator[None]:
    """Add correlation identifiers for the duration of the block."""
    merged = {**current_context(), **{k: v for k, v in values.items() if v}}
    token = _CORRELATION.set(merged)
    try:
        yield
    finally:
        _CORRELATION.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in current_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, which is what a log pipeline can actually query."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(*, level: str = "INFO", fmt: str = "json") -> None:
    """Install the root handler. Idempotent, so a re-import cannot double logs."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        JsonFormatter()
        if fmt.lower() == "json"
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    for existing in tuple(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # These two are chatty at INFO and say nothing the platform's own logs do not.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
