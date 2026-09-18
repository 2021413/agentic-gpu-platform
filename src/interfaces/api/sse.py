"""Turning domain events into Server-Sent Events (spec section 21).

Two rules drive this module.

**Never leak hidden reasoning.** The platform's agents produce chains of
thought; clients get *structured progress* only. Domain events are already
narrow, but this is the last gate before bytes leave the process, so it drops
any field known to carry model prose and truncates the rest. A leak here cannot
be taken back.

**The event id is the resumption cursor.** Only events replayed from the durable
store carry a sequence number, so only those set the SSE ``id`` field. A client
that reconnects with ``Last-Event-ID`` therefore resumes from the last *durable*
event: it may see a handful of live events twice, never a gap. Duplicates are
recoverable (events are addressed and idempotent), a silent gap is not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from sse_starlette import ServerSentEvent

from application.dto.views import EventView
from domain.events.base import DomainEvent
from domain.events.run import RunCancelled, RunCompleted, RunFailed

__all__ = [
    "SSE_PING_SECONDS",
    "TERMINAL_EVENT_NAMES",
    "live_event",
    "public_payload",
    "replayed_event",
]

SSE_PING_SECONDS: Final = 15
"""Comment frames keep proxies from closing an idle run's stream.

A planning stage can be silent for minutes; many load balancers cut a
connection after 60s of silence.
"""

TERMINAL_EVENT_NAMES: Final = frozenset(
    {RunCompleted.name, RunFailed.name, RunCancelled.name},
)
"""Which events end a stream. Taken from the domain's own event classes so a
renamed event breaks the import rather than silently hanging every client.
"""

_HIDDEN_KEYS: Final = frozenset(
    {
        "chain_of_thought",
        "completion",
        "messages",
        "prompt",
        "prompts",
        "raw_output",
        "raw_response",
        "reasoning",
        "scratchpad",
        "thinking",
        "thought",
        "tokens_text",
    }
)

_MAX_STRING_LENGTH: Final = 2000
"""Progress fields are summaries. Anything longer is model prose that escaped
into an event, and a stream is not the place to discover that.
"""

_TRUNCATION_SUFFIX: Final = "…[truncated]"


def public_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip anything a client must not see, and bound what remains."""
    return {key: _bound(value) for key, value in payload.items() if key.lower() not in _HIDDEN_KEYS}


def _bound(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STRING_LENGTH:
        return value[:_MAX_STRING_LENGTH] + _TRUNCATION_SUFFIX
    if isinstance(value, Mapping):
        return {key: _bound(item) for key, item in value.items() if key.lower() not in _HIDDEN_KEYS}
    if isinstance(value, list):
        return [_bound(item) for item in value]
    return value


def _frame(*, sequence: int | None, name: str, occurred_at: str, payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "sequence": sequence,
            "name": name,
            "occurred_at": occurred_at,
            "payload": payload,
        },
        separators=(",", ":"),
        default=str,
    )


def replayed_event(view: EventView) -> ServerSentEvent:
    """A durable, numbered event: it sets ``id`` and can be resumed from."""
    return ServerSentEvent(
        id=str(view.sequence),
        event=view.name,
        data=_frame(
            sequence=view.sequence,
            name=view.name,
            occurred_at=view.occurred_at.isoformat(),
            payload=public_payload(view.payload),
        ),
    )


def live_event(event: DomainEvent) -> ServerSentEvent:
    """An event straight off the bus: no ``id``, because it has no cursor yet."""
    return ServerSentEvent(
        event=event.name,
        data=_frame(
            sequence=None,
            name=event.name,
            occurred_at=event.occurred_at.isoformat(),
            payload=public_payload(event.payload()),
        ),
    )
