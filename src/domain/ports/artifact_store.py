"""Artifact storage port.

Patches, build logs and test reports outlive the workspace that produced them.
They are addressed by run and candidate so the API can expose them after the
workspace is gone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.value_objects.identifiers import CandidateId, RunId

__all__ = ["ArtifactRef", "ArtifactStore"]


class ArtifactRef(str):
    """Opaque storage reference. Only the store interprets it."""

    __slots__ = ()


@runtime_checkable
class ArtifactStore(Protocol):
    async def put(
        self,
        *,
        run_id: RunId,
        name: str,
        content: bytes,
        candidate_id: CandidateId | None = None,
        content_type: str = "application/octet-stream",
    ) -> ArtifactRef: ...

    async def get(self, ref: ArtifactRef) -> bytes | None: ...

    async def list_for_run(self, run_id: RunId) -> Sequence[tuple[str, ArtifactRef]]: ...

    async def delete_run(self, run_id: RunId) -> None: ...
