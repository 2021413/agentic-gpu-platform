"""Following a run in real time over Server-Sent Events (spec section 21).

The stream is the join of two sources that must not be confused:

* the **EventStore**, durable and numbered, replayed so a client that connects
  late — or reconnects — sees what it missed;
* the **EventBus**, live and unnumbered, for everything that happens next.

Ordering matters more than it looks. The subscription is opened *before* the
replay reads the store: opening it afterwards leaves a window in which an event
is published after the last stored row was read and before anyone is listening,
and that event is then lost with no way for the client to notice.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query
from sse_starlette import EventSourceResponse, ServerSentEvent

from application.use_cases.runs import GetRunUseCase, ListRunEventsUseCase
from domain.events.base import DomainEvent
from domain.ports.event_bus import EventBus
from domain.value_objects.identifiers import RunId
from interfaces.api.dependencies.providers import EventBusDep, GetRunDep, ListRunEventsDep
from interfaces.api.schemas.common import ProblemDetails
from interfaces.api.sse import (
    SSE_PING_SECONDS,
    TERMINAL_EVENT_NAMES,
    live_event,
    replayed_event,
)

__all__ = ["router"]

router = APIRouter(prefix="/v1/runs", tags=["runs"])

_REPLAY_PAGE_SIZE = 500


@router.get(
    "/{run_id}/events",
    summary="Follow a run as Server-Sent Events",
    response_class=EventSourceResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "A stream of progress events. Each frame's `id` is a resumption "
                "cursor; reconnect with `Last-Event-ID` to replay from it. The "
                "stream closes on its own when the run reaches a terminal state."
            ),
        },
        404: {"model": ProblemDetails},
    },
)
async def stream_run_events(
    run_id: UUID,
    *,
    get_run: GetRunDep,
    list_events: ListRunEventsDep,
    bus: EventBusDep,
    last_event_id: Annotated[
        int | None,
        Header(alias="Last-Event-ID", ge=0, description="Sequence to resume after."),
    ] = None,
    after: Annotated[
        int | None,
        Query(ge=0, description="Same as Last-Event-ID, for clients that cannot set headers."),
    ] = None,
) -> EventSourceResponse:
    identifier = RunId(run_id)
    # Resolved before the stream opens: once the response has started, a 404 can
    # no longer be expressed as a status code, only as a confusing empty stream.
    await get_run.execute(identifier)
    cursor = last_event_id if last_event_id is not None else after
    return EventSourceResponse(
        _run_event_stream(
            run_id=identifier,
            cursor=cursor,
            get_run=get_run,
            list_events=list_events,
            bus=bus,
        ),
        ping=SSE_PING_SECONDS,
    )


async def _run_event_stream(
    *,
    run_id: RunId,
    cursor: int | None,
    get_run: GetRunUseCase,
    list_events: ListRunEventsUseCase,
    bus: EventBus,
) -> AsyncIterator[ServerSentEvent]:
    live = bus.subscribe(run_id)
    # Scheduling the first pull *primes* the subscription: the generator runs up
    # to its first await, which is where an adapter registers with the bus. The
    # replay below then happens with a listener already attached.
    pending: asyncio.Task[DomainEvent] = asyncio.ensure_future(anext(live))
    try:
        async for frame in _replay(run_id=run_id, cursor=cursor, list_events=list_events):
            yield frame

        # Re-read after the replay: a run that finished before the client
        # connected has already emitted its terminal event into the store, so
        # the client has just received it and the stream must end rather than
        # wait forever for something that will never be published.
        detail = await get_run.execute(run_id)
        if detail.run.status.is_terminal:
            return

        while True:
            event = await pending
            pending = asyncio.ensure_future(anext(live))
            yield live_event(event)
            if event.name in TERMINAL_EVENT_NAMES:
                return
    except StopAsyncIteration:
        # The bus closed the subscription: an ordinary end of stream. A
        # client disconnect arrives as ``CancelledError`` instead and is
        # deliberately *not* caught — swallowing a cancellation inside an async
        # generator leaves the surrounding task group believing it is still
        # running.
        return
    finally:
        # Cancel the in-flight pull before closing: aclose() on a generator that
        # is still suspended inside another task raises "already running".
        if not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
        if isinstance(live, AsyncGenerator):
            await live.aclose()


async def _replay(
    *, run_id: RunId, cursor: int | None, list_events: ListRunEventsUseCase
) -> AsyncIterator[ServerSentEvent]:
    """Drain the durable history, page by page, from ``cursor`` exclusive."""
    position = cursor
    while True:
        batch = await list_events.execute(run_id, after_sequence=position, limit=_REPLAY_PAGE_SIZE)
        for view in batch:
            position = view.sequence
            yield replayed_event(view)
        if len(batch) < _REPLAY_PAGE_SIZE:
            return
