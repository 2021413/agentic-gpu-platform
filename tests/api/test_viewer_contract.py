"""The viewer's types must match the API's schemas.

Written after the viewer claimed a worker had `max_concurrency` while the API
had always called it `capacity`. TypeScript cannot catch that: the compiler
checks the client against its own declarations, and the declarations were the
thing that was wrong. Nothing failed — the number simply rendered as `NaN`.

So the declarations are checked against the generated OpenAPI schema, which is
the real contract. A field the viewer reads and the API does not serve is a
blank on screen, which is the failure mode this whole project keeps hitting:
not an error, just something quietly missing.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from domain.events.run import RepositoryContextSelected
from interfaces.api.app import create_api

CLIENT = Path("web/src/api.ts")

# Each viewer interface and the API schema it mirrors. Listed by hand: the
# mapping is a decision, and guessing it from names would be the same kind of
# assumption this test exists to refuse.
MIRRORS = {
    "Project": "ProjectResponse",
    "Toolchain": "ToolchainResponse",
    "Run": "RunResponse",
    "Candidate": "CandidateResponse",
    "CandidatePatch": "CandidatePatchResponse",
    "Review": "ReviewResponse",
    "ReviewFinding": "ReviewFindingResponse",
    "Worker": "WorkerResponse",
}

_INTERFACE = re.compile(r"export interface (\w+) \{(.*?)\n\}", re.DOTALL)
_FIELD = re.compile(r"^\s*(?:/\*\*.*?\*/\s*)?(\w+)\??:", re.MULTILINE)


def viewer_interfaces() -> dict[str, set[str]]:
    source = CLIENT.read_text(encoding="utf-8")
    found: dict[str, set[str]] = {}
    for name, body in _INTERFACE.findall(source):
        # Strip comment lines so a documented field is read once, not twice.
        cleaned = re.sub(r"/\*\*.*?\*/", "", body, flags=re.DOTALL)
        found[name] = set(_FIELD.findall(cleaned))
    return found


def schemas() -> dict[str, Any]:
    return create_api(include_internal_api=False).openapi()["components"]["schemas"]


@pytest.mark.parametrize(("interface", "schema"), sorted(MIRRORS.items()))
def test_every_field_the_viewer_reads_is_actually_served(interface: str, schema: str) -> None:
    declared = viewer_interfaces()
    assert interface in declared, f"{interface} disappeared from {CLIENT}"

    available = schemas()
    assert schema in available, f"{schema} is not in the OpenAPI document"

    served = set(available[schema].get("properties", {}))
    invented = declared[interface] - served
    assert not invented, (
        f"{interface} reads {sorted(invented)}, which {schema} does not serve. "
        f"It serves {sorted(served)}."
    )


def test_the_mirrors_cover_every_interface_the_viewer_declares() -> None:
    """A new interface must be claimed or explicitly excluded, or this test
    quietly stops covering the thing it was written for."""
    excluded = {"ContextManifest", "RunEvent"}  # event payloads, not response models
    unclaimed = set(viewer_interfaces()) - set(MIRRORS) - excluded
    assert not unclaimed, f"unchecked viewer interfaces: {sorted(unclaimed)}"


def test_the_context_manifest_matches_the_event_it_reads() -> None:
    """Not an OpenAPI schema: it arrives on the SSE stream as an event payload,
    so it is checked against the domain event's own fields."""
    served = {f.name for f in fields(RepositoryContextSelected)} - {"occurred_at", "event_id"}
    declared = viewer_interfaces()["ContextManifest"]

    invented = declared - served - {"run_id"}
    assert not invented, (
        f"the manifest reads {sorted(invented)}; the event carries {sorted(served)}"
    )
