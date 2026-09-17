"""Choosing between competing candidates (spec section 38).

Deterministic evidence first, model opinion last. No invented numeric scores:
a candidate wins because its build succeeded and its tests passed, and only
then because a reviewer preferred it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from domain.entities.candidate import Candidate
from domain.enums import ReviewVerdict

__all__ = ["CandidateSelection", "DeterministicCandidateSelectionPolicy"]


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """The chosen candidate and the evidence-based reason it won."""

    winner: Candidate | None
    rationale: str
    rejected: tuple[Candidate, ...] = ()

    @property
    def has_winner(self) -> bool:
        return self.winner is not None


def _tier(candidate: Candidate) -> tuple[int, int, int, int]:
    """Comparable evidence tier; higher is better.

    Ordering, most significant first:
      1. a reviewer PASS (only meaningful once validation already passed);
      2. tests observed passing;
      3. build observed succeeding;
      4. a non-empty patch at all.
    """
    validation = candidate.validation
    return (
        1 if candidate.review_verdict is ReviewVerdict.PASS else 0,
        1 if validation.tests_passed else 0,
        1 if validation.build_passed else 0,
        1 if candidate.patch is not None and not candidate.patch.is_empty else 0,
    )


class DeterministicCandidateSelectionPolicy:
    """Default policy: rank on tool evidence, break ties on the smallest change.

    Preferring the smaller diff among otherwise equal candidates is a
    deliberate bias towards reviewable work, not an aesthetic one.
    """

    __slots__ = ()

    name = "deterministic_evidence_first"

    def select(self, candidates: Sequence[Candidate]) -> CandidateSelection:
        usable = [c for c in candidates if not c.status.is_terminal or c.is_viable]
        if not usable:
            return CandidateSelection(None, "no candidate produced a usable patch")

        ranked = sorted(
            usable,
            key=lambda c: (
                _tier(c),
                -(c.patch.total_churn if c.patch else 0),
                -c.index,
            ),
            reverse=True,
        )
        winner = ranked[0]
        if _tier(winner)[-1] == 0:
            # Checked before viability so the rationale names the real cause: a
            # candidate whose build passed but which changed nothing is empty,
            # not broken.
            return CandidateSelection(
                None, "no candidate produced a non-empty patch", tuple(ranked)
            )
        if not winner.is_viable:
            return CandidateSelection(
                None,
                "no candidate survived deterministic validation: " + winner.validation.summary(),
                tuple(ranked),
            )
        return CandidateSelection(
            winner,
            f"candidate {winner.index} selected on {winner.validation.summary()}",
            tuple(ranked[1:]),
        )

    def viable(self, candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
        """Candidates worth sending to the reviewer at all."""
        return tuple(c for c in candidates if c.is_viable)
