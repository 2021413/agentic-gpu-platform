"""What the three agent roles are expected to produce.

These drafts are the *validated* shape of a model answer, before it becomes a
domain object. Keeping them here means the application layer never imports the
pydantic models used to validate them: the parser is a port, and its concrete
schema library stays in infrastructure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from domain.entities.review import Severity
from domain.enums import ReviewVerdict

__all__ = [
    "CodeDraft",
    "FindingDraft",
    "PlanDraft",
    "ReviewDraft",
    "TaskDraft",
    "ToolRequest",
]


@dataclass(frozen=True, slots=True)
class TaskDraft:
    key: str
    title: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    target_paths: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanDraft:
    """Planner output (spec section 6)."""

    objective: str
    tasks: tuple[TaskDraft, ...]
    assumptions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    risk_areas: tuple[str, ...] = ()
    validation_requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("a plan draft must contain at least one task")


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A tool the coder wants run before it commits to a patch."""

    tool: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CodeDraft:
    """Coder output.

    ``diff`` may be empty when the coder only wants tools executed first; the
    orchestrator then runs them and re-prompts, bounded by the iteration budget.
    """

    summary: str = ""
    diff: str = ""
    uncertainties: tuple[str, ...] = ()
    tool_requests: tuple[ToolRequest, ...] = ()
    done: bool = True

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_requests) and not self.diff.strip()


@dataclass(frozen=True, slots=True)
class FindingDraft:
    summary: str
    severity: Severity = Severity.MAJOR
    file: str | None = None
    line: int | None = None
    repair_instruction: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewDraft:
    """Reviewer output: PASS, or FAIL with actionable instructions."""

    verdict: ReviewVerdict
    summary: str = ""
    findings: tuple[FindingDraft, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict is ReviewVerdict.FAIL and not self.findings:
            raise ValueError("a FAIL verdict must carry at least one finding")

    @property
    def passed(self) -> bool:
        return self.verdict is ReviewVerdict.PASS


def format_findings(findings: Sequence[FindingDraft]) -> str:
    """Compact rendering handed back to a coder during repair."""
    return "\n".join(
        f"- [{f.severity}]"
        + (f" ({f.file}:{f.line})" if f.file else "")
        + f" {f.repair_instruction or f.summary}"
        for f in findings
    )
