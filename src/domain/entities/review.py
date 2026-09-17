"""Reviewer verdicts (spec section 6).

The reviewer sees the patch and the deterministic validation results — not the
coder's conversation. Its verdict is structured, and a FAIL must carry
actionable repair instructions or it cannot drive the repair loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique

from domain.enums import ReviewVerdict
from domain.value_objects.identifiers import CandidateId, ReviewId, RunId

__all__ = ["Review", "ReviewFinding", "Severity"]


@unique
class Severity(StrEnum):
    BLOCKER = "BLOCKER"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"

    @property
    def blocks_completion(self) -> bool:
        return self in (Severity.BLOCKER, Severity.MAJOR)


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """One defect, tied to a location when the reviewer can identify one."""

    summary: str
    severity: Severity = Severity.MAJOR
    file: str | None = None
    line: int | None = None
    repair_instruction: str | None = None

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("a finding needs a summary")


@dataclass(frozen=True, slots=True)
class Review:
    """A single review of a candidate."""

    id: ReviewId
    run_id: RunId
    candidate_id: CandidateId
    verdict: ReviewVerdict
    iteration: int
    created_at: datetime
    summary: str = ""
    findings: tuple[ReviewFinding, ...] = ()
    prompt_version: str = "v1"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("review iterations are 1-based")
        if self.verdict is ReviewVerdict.FAIL and not self.findings:
            raise ValueError("a failing review must report at least one finding")

    @property
    def passed(self) -> bool:
        return self.verdict is ReviewVerdict.PASS

    @property
    def blocking_findings(self) -> tuple[ReviewFinding, ...]:
        return tuple(f for f in self.findings if f.severity.blocks_completion)

    def repair_brief(self) -> str:
        """The instructions handed back to a coder for the repair iteration."""
        lines: list[str] = []
        for finding in self.findings:
            location = f" ({finding.file}:{finding.line})" if finding.file else ""
            instruction = finding.repair_instruction or finding.summary
            lines.append(f"- [{finding.severity}]{location} {instruction}")
        return "\n".join(lines)
