"""Worker scheduling policies (spec section 5).

Policy is pure: given the workers that exist and what a job needs, pick one.
No I/O, no registry, no HTTP — which is why every scheduling rule in this file
is unit-testable without a GPU or a network.
"""

from __future__ import annotations

from collections.abc import Sequence

from domain.entities.worker import Worker
from domain.value_objects.worker import JobRequirements

__all__ = [
    "LeastLoadedCompatibleScheduler",
    "RoundRobinScheduler",
    "eligible_workers",
]


def eligible_workers(workers: Sequence[Worker], requirements: JobRequirements) -> list[Worker]:
    """Workers that may take this job right now, in registration order."""
    return [w for w in workers if w.can_accept(requirements)]


class LeastLoadedCompatibleScheduler:
    """Default policy: the compatible worker with the most headroom.

    Ties are broken by absolute free slots, then by the longest context window
    (a roomier worker is likelier to absorb a growing conversation), then by id
    so the outcome is deterministic and reproducible in tests.
    """

    __slots__ = ()

    name = "least_loaded_compatible_worker"

    def select(
        self, *, candidates: Sequence[Worker], requirements: JobRequirements
    ) -> Worker | None:
        ranked = self.rank(candidates=candidates, requirements=requirements)
        return ranked[0] if ranked else None

    def rank(self, *, candidates: Sequence[Worker], requirements: JobRequirements) -> list[Worker]:
        return sorted(
            eligible_workers(candidates, requirements),
            key=lambda w: (
                w.utilization,
                -w.available_slots,
                -w.capabilities.context_length,
                str(w.id),
            ),
        )


class RoundRobinScheduler:
    """Even distribution, ignoring load.

    Useful when every job costs roughly the same and workers are homogeneous;
    kept in the domain to prove the port supports more than one strategy.
    """

    __slots__ = ("_cursor",)

    name = "round_robin"

    def __init__(self) -> None:
        self._cursor = 0

    def select(
        self, *, candidates: Sequence[Worker], requirements: JobRequirements
    ) -> Worker | None:
        ranked = self.rank(candidates=candidates, requirements=requirements)
        if not ranked:
            return None
        chosen = ranked[self._cursor % len(ranked)]
        self._cursor = (self._cursor + 1) % len(ranked)
        return chosen

    def rank(self, *, candidates: Sequence[Worker], requirements: JobRequirements) -> list[Worker]:
        return sorted(eligible_workers(candidates, requirements), key=lambda w: str(w.id))
