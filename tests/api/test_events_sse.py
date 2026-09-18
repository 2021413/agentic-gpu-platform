"""Following a run over Server-Sent Events."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from uuid import UUID

import httpx
import pytest
from tests.api.conftest import Harness, create_project, create_run

from domain.enums import RunStatus
from domain.events.run import RunCompleted, RunStateChanged
from domain.value_objects.identifiers import RunId
from interfaces.api.sse import public_payload

STREAM_TIMEOUT = 5.0


@dataclass(frozen=True)
class Frame:
    event: str
    data: dict[str, object]
    id: str | None


def parse_stream(text: str) -> list[Frame]:
    """Minimal SSE parser: enough to assert on ids, names and payloads.

    Frames are separated by a blank line and the wire format uses CRLF, so the
    text is normalised before splitting.
    """
    frames: list[Frame] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        if "data" in fields:
            frames.append(
                Frame(
                    event=fields.get("event", "message"),
                    data=json.loads(fields["data"]),
                    id=fields.get("id"),
                )
            )
    return frames


async def a_cancelled_run(client: httpx.AsyncClient) -> str:
    project_id = await create_project(client)
    run_id: str = (await create_run(client, project_id)).json()["id"]
    await client.post(f"/v1/runs/{run_id}/cancel", json={"reason": "enough"})
    return run_id


async def test_a_finished_run_replays_its_history_and_closes(
    client: httpx.AsyncClient,
) -> None:
    """No live subscription can help a run that is already over: it must end."""
    run_id = await a_cancelled_run(client)

    response = await asyncio.wait_for(
        client.get(f"/v1/runs/{run_id}/events"), timeout=STREAM_TIMEOUT
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = parse_stream(response.text)
    names = [frame.event for frame in frames]
    assert names[0] == "run.created"
    assert "run.cancelled" in names
    assert all(frame.data["sequence"] == int(frame.id or "0") for frame in frames)


async def test_replayed_frames_carry_the_resumption_cursor(
    client: httpx.AsyncClient,
) -> None:
    run_id = await a_cancelled_run(client)

    full = await asyncio.wait_for(client.get(f"/v1/runs/{run_id}/events"), timeout=STREAM_TIMEOUT)
    frames = parse_stream(full.text)
    first_id = frames[0].id
    assert first_id is not None

    resumed = await asyncio.wait_for(
        client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": first_id}),
        timeout=STREAM_TIMEOUT,
    )
    resumed_frames = parse_stream(resumed.text)

    assert [frame.event for frame in resumed_frames] == [f.event for f in frames[1:]]
    assert "run.created" not in [frame.event for frame in resumed_frames]


async def test_a_query_parameter_can_replace_the_header(client: httpx.AsyncClient) -> None:
    """Some clients (curl pipelines, browsers' EventSource) cannot set headers."""
    run_id = await a_cancelled_run(client)

    response = await asyncio.wait_for(
        client.get(f"/v1/runs/{run_id}/events", params={"after": 1}),
        timeout=STREAM_TIMEOUT,
    )

    assert "run.created" not in [frame.event for frame in parse_stream(response.text)]


async def test_live_events_are_streamed_until_the_run_ends(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    """An open run switches to the bus, and the terminal event closes the stream."""
    project_id = await create_project(client)
    run_id = (await create_run(client, project_id)).json()["id"]
    identifier = RunId(UUID(run_id))

    stream = asyncio.ensure_future(client.get(f"/v1/runs/{run_id}/events"))
    await asyncio.sleep(0.05)  # let the handler subscribe before anything is published

    now = harness.clock.now()
    await harness.bus.publish(
        [
            RunStateChanged(
                occurred_at=now,
                run_id=identifier,
                previous=RunStatus.CREATED,
                current=RunStatus.PLANNING,
            ),
            RunCompleted(occurred_at=now, run_id=identifier),
        ]
    )

    response = await asyncio.wait_for(stream, timeout=STREAM_TIMEOUT)
    frames = parse_stream(response.text)
    names = [frame.event for frame in frames]

    assert names[-1] == "run.completed"
    assert "run.state_changed" in names
    live = [frame for frame in frames if frame.event == "run.completed"]
    assert live[0].id is None, "a live event has no durable sequence yet"
    assert live[0].data["sequence"] is None


async def test_events_of_another_run_are_not_delivered(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    project_id = await create_project(client)
    watched = (await create_run(client, project_id, idempotency_key="watched")).json()["id"]
    other = (await create_run(client, project_id, idempotency_key="other")).json()["id"]

    stream = asyncio.ensure_future(client.get(f"/v1/runs/{watched}/events"))
    await asyncio.sleep(0.05)

    now = harness.clock.now()
    await harness.bus.publish([RunCompleted(occurred_at=now, run_id=RunId(UUID(other)))])
    await harness.bus.publish([RunCompleted(occurred_at=now, run_id=RunId(UUID(watched)))])

    response = await asyncio.wait_for(stream, timeout=STREAM_TIMEOUT)
    frames = [frame for frame in parse_stream(response.text) if frame.event == "run.completed"]

    assert len(frames) == 1
    assert frames[0].data["payload"]["run_id"] == watched


async def test_streaming_an_unknown_run_is_a_problem_document(
    client: httpx.AsyncClient,
) -> None:
    """The 404 is decided before the stream opens, while a status can still be sent."""
    response = await client.get("/v1/runs/6b8f4c5e-0000-4000-8000-0000000000cc/events")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize("hidden", ["reasoning", "chain_of_thought", "raw_output", "prompt"])
def test_hidden_reasoning_never_reaches_a_frame(hidden: str) -> None:
    payload = {hidden: "first I will think about...", "status": "PLANNING"}

    assert public_payload(payload) == {"status": "PLANNING"}


def test_long_prose_is_truncated_rather_than_streamed_whole() -> None:
    """A field that grew into model prose is a leak in progress; bound it."""
    rendered = public_payload({"reason": "x" * 5000})

    assert isinstance(rendered["reason"], str)
    assert len(rendered["reason"]) < 5000
    assert rendered["reason"].endswith("[truncated]")


def test_nested_hidden_fields_are_stripped_too() -> None:
    rendered = public_payload({"detail": {"thinking": "...", "tasks": 3}})

    assert rendered == {"detail": {"tasks": 3}}
