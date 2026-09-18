"""Deterministic evidence decides, never an invented score (spec section 38)."""

from __future__ import annotations

from datetime import datetime

from domain.entities.candidate import Candidate
from domain.enums import ReviewVerdict
from domain.services.candidate_selection import DeterministicCandidateSelectionPolicy
from domain.value_objects.identifiers import CandidateId, RunId
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolKind, ToolResult
from domain.value_objects.validation import ValidationReport

RUN = RunId.generate()


def build(exit_code: int) -> ToolResult:
    return ToolResult("make", ToolKind.BUILD, "make", exit_code)


def suite(exit_code: int) -> ToolResult:
    return ToolResult("pytest", ToolKind.TEST, "pytest", exit_code)


def make_candidate(
    now: datetime,
    index: int,
    *,
    results: tuple[ToolResult, ...] = (),
    churn: int = 1,
    verdict: ReviewVerdict | None = None,
    empty_patch: bool = False,
) -> Candidate:
    candidate = Candidate.create(
        candidate_id=CandidateId.generate(), run_id=RUN, index=index, now=now
    )
    candidate.start_coding(now=now)
    diff = "" if empty_patch else "diff --git a/x.py b/x.py\n" + "+line\n" * churn
    candidate.submit_patch(patch=Patch.from_unified_diff(diff), now=now)
    candidate.start_validation(now)
    candidate.record_validation(report=ValidationReport(results), now=now)
    if verdict is not None:
        candidate.record_review(verdict)
    return candidate


def test_a_building_and_passing_candidate_beats_a_failing_one(now: datetime) -> None:
    good = make_candidate(now, 0, results=(build(0), suite(0)))
    bad = make_candidate(now, 1, results=(build(0), suite(1)))

    selection = DeterministicCandidateSelectionPolicy().select([bad, good])
    assert selection.winner is good
    assert "tests=pass" in selection.rationale


def test_a_failing_build_disqualifies(now: datetime) -> None:
    broken = make_candidate(now, 0, results=(build(1),))
    assert DeterministicCandidateSelectionPolicy().select([broken]).winner is None


def test_an_empty_patch_never_wins(now: datetime) -> None:
    empty = make_candidate(now, 0, results=(build(0), suite(0)), empty_patch=True)
    selection = DeterministicCandidateSelectionPolicy().select([empty])
    assert selection.winner is None
    assert "non-empty patch" in selection.rationale


def test_a_reviewer_pass_outranks_equal_evidence(now: datetime) -> None:
    reviewed = make_candidate(now, 0, results=(build(0), suite(0)), verdict=ReviewVerdict.PASS)
    unreviewed = make_candidate(now, 1, results=(build(0), suite(0)))
    assert DeterministicCandidateSelectionPolicy().select([unreviewed, reviewed]).winner is reviewed


def test_the_smaller_diff_breaks_a_tie(now: datetime) -> None:
    small = make_candidate(now, 0, results=(build(0), suite(0)), churn=2)
    large = make_candidate(now, 1, results=(build(0), suite(0)), churn=50)
    assert DeterministicCandidateSelectionPolicy().select([large, small]).winner is small


def test_unvalidated_candidates_are_still_comparable(now: datetime) -> None:
    """A run with no test command must still be able to pick a winner."""
    only_built = make_candidate(now, 0, results=(build(0),))
    assert DeterministicCandidateSelectionPolicy().select([only_built]).winner is only_built


def test_no_candidate_yields_no_winner() -> None:
    selection = DeterministicCandidateSelectionPolicy().select([])
    assert selection.winner is None
    assert not selection.has_winner


def test_viable_filters_what_reaches_the_reviewer(now: datetime) -> None:
    good = make_candidate(now, 0, results=(build(0), suite(0)))
    bad = make_candidate(now, 1, results=(build(1),))
    assert DeterministicCandidateSelectionPolicy().viable([good, bad]) == (good,)
