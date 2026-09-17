"""Deterministic validation outcome for one candidate.

This is the evidence the candidate-selection policy and the reviewer rely on.
It is assembled exclusively from ``ToolResult`` values, never from model prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from domain.enums import FailureKind
from domain.value_objects.tools import ToolKind, ToolResult

__all__ = ["ValidationReport"]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregated build / test / static-analysis results for one candidate."""

    results: tuple[ToolResult, ...] = ()
    static_analysis_is_blocking: bool = False

    def _of_kind(self, kind: ToolKind) -> tuple[ToolResult, ...]:
        return tuple(r for r in self.results if r.kind is kind)

    @property
    def build_results(self) -> tuple[ToolResult, ...]:
        return self._of_kind(ToolKind.BUILD)

    @property
    def test_results(self) -> tuple[ToolResult, ...]:
        return self._of_kind(ToolKind.TEST)

    @property
    def static_analysis_results(self) -> tuple[ToolResult, ...]:
        return self._of_kind(ToolKind.STATIC_ANALYSIS)

    @staticmethod
    def _all_passed(results: Sequence[ToolResult]) -> bool | None:
        """``None`` when the stage did not run: unknown is not the same as failed."""
        if not results:
            return None
        return all(r.succeeded for r in results)

    @property
    def build_passed(self) -> bool | None:
        return self._all_passed(self.build_results)

    @property
    def tests_passed(self) -> bool | None:
        return self._all_passed(self.test_results)

    @property
    def static_analysis_passed(self) -> bool | None:
        return self._all_passed(self.static_analysis_results)

    @property
    def is_viable(self) -> bool:
        """A candidate is viable when nothing that ran proves it broken.

        Stages that did not run cannot disqualify a candidate; only observed
        failures can. Static analysis is advisory unless configured otherwise.
        """
        if self.build_passed is False or self.tests_passed is False:
            return False
        return not (self.static_analysis_is_blocking and self.static_analysis_passed is False)

    @property
    def failure_kind(self) -> FailureKind | None:
        """Why the candidate is not viable, for the retry policy to branch on."""
        if self.build_passed is False:
            return FailureKind.COMPILATION
        if self.tests_passed is False:
            return FailureKind.TEST
        if self.static_analysis_is_blocking and self.static_analysis_passed is False:
            return FailureKind.TEST
        return None

    @property
    def failing_results(self) -> tuple[ToolResult, ...]:
        return tuple(r for r in self.results if not r.succeeded)

    def extended_with(self, results: Sequence[ToolResult]) -> ValidationReport:
        return replace(self, results=self.results + tuple(results))

    def summary(self) -> str:
        def render(label: str, value: bool | None) -> str:
            return f"{label}={'skipped' if value is None else ('pass' if value else 'fail')}"

        return " ".join(
            (
                render("build", self.build_passed),
                render("tests", self.tests_passed),
                render("static_analysis", self.static_analysis_passed),
            )
        )


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    """Which deterministic stages to run for a candidate."""

    build: bool = True
    tests: bool = True
    static_analysis: bool = False
    extra_commands: tuple[str, ...] = field(default=())
