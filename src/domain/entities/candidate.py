"""One competing implementation attempt (spec sections 9 and 38).

Each candidate owns an isolated workspace and produces a patch. Candidates are
compared on deterministic evidence first; the reviewer only arbitrates between
candidates that already build and pass their tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from domain.entities.base import Entity
from domain.enums import CandidateStatus, FailureKind, ReviewVerdict
from domain.events.run import (
    CandidateCompleted,
    CandidateStarted,
    ValidationCompleted,
    ValidationStarted,
)
from domain.exceptions import CandidateError, InvalidStateTransitionError
from domain.value_objects.identifiers import CandidateId, JobId, RunId, WorkerId, WorkspaceId
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolResult
from domain.value_objects.validation import ValidationReport

__all__ = ["Candidate"]


class Candidate(Entity):
    """A single best-of-N attempt at satisfying the run objective."""

    __slots__ = (
        "_coder_iterations",
        "_completed_at",
        "_created_at",
        "_failure_kind",
        "_failure_reason",
        "_id",
        "_index",
        "_last_job_id",
        "_patch",
        "_repair_iterations",
        "_review_verdict",
        "_run_id",
        "_started_at",
        "_status",
        "_summary",
        "_uncertainties",
        "_validation",
        "_worker_id",
        "_workspace_id",
    )

    def __init__(
        self,
        *,
        candidate_id: CandidateId,
        run_id: RunId,
        index: int,
        created_at: datetime,
        workspace_id: WorkspaceId | None = None,
        status: CandidateStatus = CandidateStatus.CREATED,
        patch: Patch | None = None,
        validation: ValidationReport | None = None,
        summary: str = "",
        uncertainties: tuple[str, ...] = (),
        coder_iterations: int = 0,
        repair_iterations: int = 0,
        worker_id: WorkerId | None = None,
        last_job_id: JobId | None = None,
        review_verdict: ReviewVerdict | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        failure_kind: FailureKind | None = None,
        failure_reason: str | None = None,
    ) -> None:
        super().__init__()
        if index < 0:
            raise ValueError("candidate index must not be negative")
        self._id = candidate_id
        self._run_id = run_id
        self._index = index
        self._created_at = created_at
        self._workspace_id = workspace_id
        self._status = status
        self._patch = patch
        self._validation = validation or ValidationReport()
        self._summary = summary
        self._uncertainties = uncertainties
        self._coder_iterations = coder_iterations
        self._repair_iterations = repair_iterations
        self._worker_id = worker_id
        self._last_job_id = last_job_id
        self._review_verdict = review_verdict
        self._started_at = started_at
        self._completed_at = completed_at
        self._failure_kind = failure_kind
        self._failure_reason = failure_reason

    @classmethod
    def create(
        cls, *, candidate_id: CandidateId, run_id: RunId, index: int, now: datetime
    ) -> Candidate:
        return cls(candidate_id=candidate_id, run_id=run_id, index=index, created_at=now)

    # -- accessors ------------------------------------------------------
    @property
    def identity(self) -> CandidateId:
        return self._id

    @property
    def id(self) -> CandidateId:
        return self._id

    @property
    def run_id(self) -> RunId:
        return self._run_id

    @property
    def index(self) -> int:
        return self._index

    @property
    def status(self) -> CandidateStatus:
        return self._status

    @property
    def workspace_id(self) -> WorkspaceId | None:
        return self._workspace_id

    @property
    def patch(self) -> Patch | None:
        return self._patch

    @property
    def validation(self) -> ValidationReport:
        return self._validation

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def uncertainties(self) -> tuple[str, ...]:
        return self._uncertainties

    @property
    def coder_iterations(self) -> int:
        return self._coder_iterations

    @property
    def repair_iterations(self) -> int:
        return self._repair_iterations

    @property
    def worker_id(self) -> WorkerId | None:
        return self._worker_id

    @property
    def last_job_id(self) -> JobId | None:
        return self._last_job_id

    @property
    def review_verdict(self) -> ReviewVerdict | None:
        return self._review_verdict

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at

    @property
    def failure_kind(self) -> FailureKind | None:
        return self._failure_kind

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def is_viable(self) -> bool:
        """Has a non-empty patch and no deterministic evidence against it."""
        return self._patch is not None and not self._patch.is_empty and self._validation.is_viable

    # -- lifecycle ------------------------------------------------------
    def attach_workspace(self, workspace_id: WorkspaceId) -> None:
        if self._workspace_id is not None and self._workspace_id != workspace_id:
            raise CandidateError(
                "candidate already owns a workspace",
                candidate_id=str(self._id),
                workspace_id=str(self._workspace_id),
            )
        self._workspace_id = workspace_id

    def start_coding(self, *, now: datetime, job_id: JobId | None = None) -> None:
        if self._status.is_terminal:
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.CODING)
        self._status = CandidateStatus.CODING
        self._coder_iterations += 1
        self._started_at = self._started_at or now
        self._last_job_id = job_id
        self.record(
            CandidateStarted(
                occurred_at=now,
                run_id=self._run_id,
                candidate_id=self._id,
                index=self._index,
                job_id=job_id,
            )
        )

    def record_worker(self, worker_id: WorkerId) -> None:
        """Remember where the last inference ran, for observability and affinity."""
        self._worker_id = worker_id

    def submit_patch(
        self, *, patch: Patch, now: datetime, summary: str = "", uncertainties: tuple[str, ...] = ()
    ) -> None:
        del now  # recorded on completion, not on submission
        if self._status is not CandidateStatus.CODING:
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.VALIDATING)
        self._patch = patch
        self._summary = summary
        self._uncertainties = uncertainties

    def start_validation(self, now: datetime) -> None:
        if self._status not in (CandidateStatus.CODING, CandidateStatus.VALIDATING):
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.VALIDATING)
        self._status = CandidateStatus.VALIDATING
        self.record(ValidationStarted(occurred_at=now, run_id=self._run_id, candidate_id=self._id))

    def append_validation(self, *, results: Sequence[ToolResult], now: datetime) -> None:
        """Accumulate the results of one deterministic stage.

        Validation is several jobs — build, then tests, then static analysis —
        so the report grows stage by stage and is only closed by
        ``record_validation``. Appending keeps the candidate in VALIDATING.
        """
        del now  # the timestamps live on the tool results themselves
        if self._status is not CandidateStatus.VALIDATING:
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.VALIDATING)
        self._validation = self._validation.extended_with(results)

    def record_validation(
        self,
        *,
        now: datetime,
        report: ValidationReport | None = None,
        static_analysis_is_blocking: bool = False,
    ) -> None:
        """Close validation on the accumulated evidence."""
        if self._status is not CandidateStatus.VALIDATING:
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.VALIDATED)
        final = report if report is not None else self._validation
        self._validation = replace(final, static_analysis_is_blocking=static_analysis_is_blocking)
        self._status = CandidateStatus.VALIDATED
        self.record(
            ValidationCompleted(
                occurred_at=now,
                run_id=self._run_id,
                candidate_id=self._id,
                viable=self._validation.is_viable,
                summary=self._validation.summary(),
            )
        )

    def record_review(self, verdict: ReviewVerdict) -> None:
        self._review_verdict = verdict

    def start_repair(self, now: datetime) -> None:
        """Send the candidate back to the coder with reviewer instructions."""
        if self._status.is_terminal:
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.CODING)
        self._repair_iterations += 1
        self._status = CandidateStatus.CODING
        self._coder_iterations += 1
        self.record(
            CandidateStarted(
                occurred_at=now,
                run_id=self._run_id,
                candidate_id=self._id,
                index=self._index,
            )
        )

    def select(self, now: datetime) -> None:
        if self._status is CandidateStatus.SELECTED:
            return
        if self._status.is_terminal:
            raise InvalidStateTransitionError("Candidate", self._status, CandidateStatus.SELECTED)
        self._finish(CandidateStatus.SELECTED, now)

    def reject(self, *, now: datetime, reason: str) -> None:
        """Discard a candidate that lost to a better one."""
        if self._status.is_terminal:
            return
        self._failure_reason = reason
        self._finish(CandidateStatus.REJECTED, now)

    def fail(self, *, now: datetime, kind: FailureKind, reason: str) -> None:
        if self._status.is_terminal:
            return
        self._failure_kind = kind
        self._failure_reason = reason
        self._finish(CandidateStatus.FAILED, now)

    def cancel(self, now: datetime) -> None:
        if self._status.is_terminal:
            return
        self._failure_kind = FailureKind.CANCELLED
        self._finish(CandidateStatus.CANCELLED, now)

    def _finish(self, status: CandidateStatus, now: datetime) -> None:
        self._status = status
        self._completed_at = now
        self.record(
            CandidateCompleted(
                occurred_at=now,
                run_id=self._run_id,
                candidate_id=self._id,
                status=status,
                changed_files=len(self._patch.files) if self._patch else 0,
            )
        )

    def __repr__(self) -> str:
        return f"Candidate(index={self._index}, status={self._status}, viable={self.is_viable})"
