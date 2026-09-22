"""The context manifest must survive the event log.

The viewer reads it back from storage as well as from the live stream, so a
payload that renders out but cannot be loaded in would show a full selection
while a run is going and nothing at all afterwards.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from domain.enums import AgentRole
from domain.events.run import RepositoryContextSelected
from domain.value_objects.identifiers import CandidateId, RunId
from infrastructure.database.event_codec import load_event

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def an_event() -> RepositoryContextSelected:
    return RepositoryContextSelected(
        occurred_at=NOW,
        event_id=uuid4(),
        run_id=RunId.generate(),
        role=AgentRole.CODER,
        candidate_id=CandidateId.generate(),
        files={"src/client/client_network.c": 1204, "Makefile": 118},
        tree=("Makefile", "src/client/client_network.c", "bin/client"),
        notes=("14 relevant file(s) omitted: max_files=40, max_tokens=12288",),
        estimated_tokens=1322,
        budget_tokens=12288,
    )


def test_the_manifest_round_trips_through_storage() -> None:
    original = an_event()

    restored = load_event(
        name=original.name,
        payload=dict(original.payload()),
        occurred_at=NOW,
        event_id=original.event_id,
    )

    assert isinstance(restored, RepositoryContextSelected)
    assert restored.files == original.files
    assert restored.tree == original.tree
    assert restored.notes == original.notes
    assert restored.estimated_tokens == 1322
    assert restored.budget_tokens == 12288
    assert restored.role is AgentRole.CODER
    assert restored.candidate_id == original.candidate_id
    assert restored.run_id == original.run_id


def test_an_empty_selection_survives_as_an_empty_selection() -> None:
    """The shape of the defect: no files at all. It must read back as empty,
    never as missing, or the viewer cannot tell "nothing chosen" from
    "nothing recorded"."""
    event = RepositoryContextSelected(
        occurred_at=NOW,
        event_id=uuid4(),
        run_id=RunId.generate(),
        role=AgentRole.PLANNER,
        notes=("no file matched the request; only the tree is provided",),
        budget_tokens=12288,
    )

    restored = load_event(
        name=event.name,
        payload=dict(event.payload()),
        occurred_at=NOW,
        event_id=event.event_id,
    )

    assert isinstance(restored, RepositoryContextSelected)
    assert restored.files == {}
    assert restored.candidate_id is None
    assert "no file matched" in restored.notes[0]
