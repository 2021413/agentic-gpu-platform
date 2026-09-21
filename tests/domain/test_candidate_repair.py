"""What a repair must forget (spec sections 9 and 38).

A repair replaces the code. Anything measured against the previous code is no
longer evidence about this candidate, and treating it as evidence is not a
cosmetic slip: it decides whether the run ships or fails.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from domain.entities.candidate import Candidate
from domain.enums import CandidateStatus
from domain.exceptions import InvalidStateTransitionError
from domain.value_objects.identifiers import CandidateId, RunId
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolKind, ToolResult
from domain.value_objects.validation import ValidationReport

NOW = datetime(2026, 1, 1, tzinfo=UTC)
DIFF = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+new\n"


def failing_suite() -> ToolResult:
    return ToolResult("pytest", ToolKind.TEST, "pytest", 1)


def passing_build() -> ToolResult:
    return ToolResult("make", ToolKind.BUILD, "make", 0)


def validated_candidate(
    *, results: tuple[ToolResult, ...], static_analysis_is_blocking: bool = False
) -> Candidate:
    candidate = Candidate.create(
        candidate_id=CandidateId.generate(), run_id=RunId.generate(), index=0, now=NOW
    )
    candidate.start_coding(now=NOW)
    candidate.submit_patch(patch=Patch.from_unified_diff(DIFF), now=NOW)
    candidate.start_validation(NOW)
    candidate.record_validation(
        report=ValidationReport(results),
        now=NOW,
        static_analysis_is_blocking=static_analysis_is_blocking,
    )
    return candidate


def test_a_repair_discards_the_results_of_the_code_it_replaced() -> None:
    """The bug this exists for.

    The stale report made the orchestrator find every validation stage already
    recorded, skip the second round entirely, and reject the repaired candidate
    for the failure the repair had just fixed.
    """
    candidate = validated_candidate(results=(passing_build(), failing_suite()))
    assert candidate.validation.tests_passed is False

    candidate.start_repair(NOW)

    assert candidate.status is CandidateStatus.CODING
    assert candidate.validation.results == ()
    # "did not run" and "failed" are different answers, and only the first one
    # makes the orchestrator schedule the stage again.
    assert candidate.validation.tests_passed is None
    assert candidate.validation.build_passed is None


def test_a_repair_keeps_the_policy_it_will_be_judged_by() -> None:
    """Clearing results must not quietly relax how they are read."""
    candidate = validated_candidate(results=(failing_suite(),), static_analysis_is_blocking=True)
    assert candidate.validation.static_analysis_is_blocking is True

    candidate.start_repair(NOW)

    assert candidate.validation.static_analysis_is_blocking is True


def test_a_repairing_candidate_is_not_eligible_for_selection() -> None:
    """What actually keeps a half-repaired candidate out of the running.

    Not ``is_viable``: an empty report is viable on purpose, because a stage
    that did not run must not disqualify anything — that is what lets a project
    with no build or test command finish at all. The guard is the status, so
    that is what this asserts.
    """
    candidate = validated_candidate(results=(passing_build(),))
    candidate.start_repair(NOW)

    assert candidate.status is CandidateStatus.CODING
    assert candidate.status is not CandidateStatus.VALIDATED
    # An empty report is deliberately "nothing against it"; the code below is
    # the contract that keeps that from meaning "ready to ship".
    assert candidate.validation.is_viable is True


def test_the_repair_counter_still_advances() -> None:
    """The budget is what stops an endless loop; clearing results must not
    clear the count of how many times we have been round."""
    candidate = validated_candidate(results=(failing_suite(),))
    before = candidate.repair_iterations

    candidate.start_repair(NOW)

    assert candidate.repair_iterations == before + 1
    assert candidate.coder_iterations >= 2


def test_a_finished_candidate_cannot_be_repaired() -> None:
    candidate = validated_candidate(results=(failing_suite(),))
    candidate.reject(now=NOW, reason="lost to a better candidate")

    with pytest.raises(InvalidStateTransitionError):
        candidate.start_repair(NOW)
