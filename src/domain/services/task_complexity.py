"""Deciding whether an objective needs a planner (spec section 37).

A separate routing model is explicitly *not* required in v1: a deterministic
heuristic is cheaper, reproducible, and good enough to skip planning on
genuinely small work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum, unique

__all__ = ["HeuristicTaskComplexityPolicy", "TaskComplexity", "TaskComplexityPolicy"]


@unique
class TaskComplexity(StrEnum):
    TRIVIAL = "TRIVIAL"
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"

    @property
    def requires_plan(self) -> bool:
        return self is not TaskComplexity.TRIVIAL


@dataclass(frozen=True, slots=True)
class ComplexityAssessment:
    complexity: TaskComplexity
    recommended_candidates: int
    rationale: str


class TaskComplexityPolicy:
    """Port-shaped base so an LLM-based router can replace the heuristic later."""

    __slots__ = ()

    def assess(self, objective: str, *, max_candidates: int = 3) -> ComplexityAssessment:
        raise NotImplementedError


_MULTI_STEP = re.compile(
    r"\b(and then|refactor|migrat|architect|redesign|across|end[- ]to[- ]end|"
    r"integrat|rewrite|multiple|several)\b",
    re.IGNORECASE,
)
_TRIVIAL = re.compile(
    r"\b(typo|rename|comment|docstring|bump|format|whitespace|changelog)\b", re.IGNORECASE
)


class HeuristicTaskComplexityPolicy(TaskComplexityPolicy):
    """Deterministic assessment from the shape of the objective.

    Signals: explicit multi-step vocabulary, objective length, and how many
    files or requirements the text enumerates. Cheap, explainable, and never
    silently wrong in a way a user cannot see.
    """

    __slots__ = ("_complex_length", "_trivial_length")

    def __init__(self, *, trivial_length: int = 80, complex_length: int = 320) -> None:
        self._trivial_length = trivial_length
        self._complex_length = complex_length

    def assess(self, objective: str, *, max_candidates: int = 3) -> ComplexityAssessment:
        text = objective.strip()
        bullet_count = len(re.findall(r"(?m)^\s*[-*\d]+[.)]?\s+", text))
        path_count = len(re.findall(r"[\w/]+\.\w{1,5}\b", text))

        if _TRIVIAL.search(text) and len(text) <= self._trivial_length:
            return ComplexityAssessment(
                TaskComplexity.TRIVIAL, 1, "short, mechanical objective: planner skipped"
            )

        if (
            _MULTI_STEP.search(text)
            or len(text) >= self._complex_length
            or bullet_count >= 3
            or path_count >= 4
        ):
            return ComplexityAssessment(
                TaskComplexity.COMPLEX,
                min(max(2, max_candidates), max_candidates),
                "multi-step objective: plan first and try several candidates",
            )

        return ComplexityAssessment(
            TaskComplexity.SIMPLE, 1, "single-step objective: plan once, one candidate"
        )
