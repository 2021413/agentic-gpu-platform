"""The run orchestrator (spec sections 2, 9, 10 and 37).

This is the brain, and it is deliberately boring: a governed state machine that
schedules jobs, consumes their results, and decides the next step. It never
calls a GPU directly, never runs a shell command, never writes to the database
outside a unit of work. Every one of those is reached through a port, which is
why the whole workflow runs in tests with no GPU, no Docker and no network.

Two rules shape everything here:

* only deterministic tool results may establish that something works;
* every loop is bounded, and an exhausted budget is a normal outcome that ends
  the run cleanly rather than an exception nobody catches.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from application.dto.agent_io import CodeDraft, PlanDraft, ReviewDraft, format_findings
from application.orchestration.agents import CoderAgent, PlannerAgent, ReviewerAgent
from application.orchestration.worker_pool import WorkerPool
from application.ports import RunCoordinator, ToolExecutorFactory, UnitOfWorkFactory
from application.services.event_publisher import commit_and_publish
from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan, PlanTask
from domain.entities.project import Project
from domain.entities.review import Review, ReviewFinding
from domain.entities.run import Run
from domain.enums import (
    AgentRole,
    CandidateStatus,
    FailureKind,
    JobType,
    Priority,
    ReviewVerdict,
    RunStatus,
)
from domain.events.run import RepositoryContextSelected
from domain.exceptions import (
    DomainError,
    EntityNotFoundError,
    InferenceError,
    LLMTimeoutError,
    NoCompatibleWorkerError,
    OutputTruncatedError,
    RunNotModifiableError,
    StructuredOutputError,
    ToolExecutionError,
    WorkspaceError,
)
from domain.ports.clock import Clock, IdGenerator
from domain.ports.event_bus import EventBus
from domain.ports.job_queue import JobQueue
from domain.ports.repositories import UnitOfWork
from domain.ports.repository_context import (
    ContextRequest,
    RepositoryContext,
    RepositoryContextProvider,
)
from domain.ports.workspace import WorkspaceManager
from domain.services.candidate_selection import DeterministicCandidateSelectionPolicy
from domain.services.retry_policy import RetryPolicy
from domain.value_objects.identifiers import (
    CandidateId,
    JobId,
    PlanId,
    ReviewId,
    RunId,
    TaskId,
)
from domain.value_objects.lease import Lease
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ExecutionLimits, ToolInvocation, ToolKind, ToolResult
from domain.value_objects.validation import ValidationReport
from domain.value_objects.worker import JobRequirements
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceRole

__all__ = ["OrchestratorConfig", "RunOrchestrator"]

_log = logging.getLogger(__name__)

# Deterministic validation runs as a sequence of jobs so each stage is observable
# and individually retryable, rather than one opaque "validate" step.
_VALIDATION_STAGES: tuple[JobType, ...] = (
    JobType.BUILD,
    JobType.TEST,
    JobType.STATIC_ANALYSIS,
)

# The tool for a stage exists only when the project configures that command,
# which is why a missing tool means "skipped" rather than "failed".
_STAGE_TOOLS: dict[JobType, tuple[str, ToolKind]] = {
    JobType.BUILD: ("build", ToolKind.BUILD),
    JobType.TEST: ("run_tests", ToolKind.TEST),
    JobType.STATIC_ANALYSIS: ("static_analysis", ToolKind.STATIC_ANALYSIS),
}


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    lease_duration: timedelta = timedelta(seconds=120)
    job_max_attempts: int = 3
    context_max_files: int = 40
    context_max_tokens: int = 24_000
    """Ceiling on the repository view. The fleet lowers it; nothing raises it."""
    prompt_overhead_tokens: int = 2_048
    """Room kept in the window for the prompt that is not the repository view.

    The fleet reports how large a *prompt* it can take. That number was handed
    straight to the context provider as the budget for the code excerpt, which
    silently assumed the excerpt was the whole prompt. It is not: the system
    instructions, the objective, the plan, the accumulated review findings and
    the JSON schema all ride in the same window and none of them were counted.

    The first real run against a 32768-token engine failed by exactly one token:

        maximum context length is 32768 tokens. However, you requested 4096
        output tokens and your prompt contains at least 28673 input tokens

    28672 is 32768 - 4096 to the token. The excerpt had been packed to fill the
    entire prompt budget, and the template pushed it over. Off by one, but the
    cost is a whole cold GPU and a run that reaches the coder and dies.

    A fixed allowance rather than a measurement, because the request is sized
    before the prompt is rendered. It is deliberately generous: over-reserving
    costs a few files of context, under-reserving costs the run.
    """
    reserved_output_tokens: int = 4_096
    """Room kept in the window for the answer.

    Sized for the coder, which is the only role that writes whole files back —
    a diff of a few lines costs a fraction of this, a rewritten module does
    not. Too small and the reply is truncated mid-JSON, which reaches the
    repair loop as malformed JSON rather than as "you ran out of room".
    """
    def excerpt_budget(self, fleet_budget: int | None) -> int:
        """How many tokens of repository the prompt may carry.

        `fleet_budget` is room for the WHOLE prompt, so the part of the prompt
        that is not code has to come out of it first. The fleet only ever
        lowers the configured ceiling: an empty fleet leaves it alone, because
        there is nothing to learn from and guessing is what caused the
        24000-against-16384 mismatch in the first place.
        """
        if fleet_budget is None:
            return self.context_max_tokens
        return min(self.context_max_tokens, max(0, fleet_budget - self.prompt_overhead_tokens))

    validation_timeout_seconds: float = 900.0
    static_analysis_is_blocking: bool = False
    integrate_on_success: bool = True
    require_approval: bool = False
    """Hold a reviewed run until a human lets it land.

    Off by default, and deliberately so: integration writes into someone else's
    repository, but a run waiting on an approval nobody is watching is a run
    that never finishes. Turning this on is a statement that somebody is.
    """


class RunOrchestrator:
    """Drives runs from objective to a deterministic COMPLETED or FAILED."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        bus: EventBus,
        queue: JobQueue,
        pool: WorkerPool,
        coordinator: RunCoordinator,
        workspaces: WorkspaceManager,
        tools: ToolExecutorFactory,
        context: RepositoryContextProvider,
        planner: PlannerAgent,
        coder: CoderAgent,
        reviewer: ReviewerAgent,
        clock: Clock,
        ids: IdGenerator,
        retry_policy: RetryPolicy | None = None,
        selection: DeterministicCandidateSelectionPolicy | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._bus = bus
        self._queue = queue
        self._pool = pool
        self._coordinator = coordinator
        self._workspaces = workspaces
        self._tools = tools
        self._context = context
        self._planner = planner
        self._coder = coder
        self._reviewer = reviewer
        self._clock = clock
        self._ids = ids
        self._retry = retry_policy or RetryPolicy()
        self._selection = selection or DeterministicCandidateSelectionPolicy()
        self._config = config or OrchestratorConfig()

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------
    async def start(self, run_id: RunId) -> None:
        """Schedule the first stage of a freshly created run.

        Whether the planner is involved was decided when the run was created, by
        an explicit complexity policy; the orchestrator only obeys it.
        """
        async with self._coordinator.lock(run_id):
            async with self._uow_factory() as uow:
                run = await self._require_run(uow, run_id)
                if run.status is not RunStatus.CREATED:
                    return  # already started: at-least-once delivery, not an error
                project = await self._require_project(uow, run)

                jobs = await self._open_first_stage(uow, run=run, project=project)

                await self._persist_jobs(uow, jobs)
                await uow.runs.update(run)
                uow.collect(run)
                await commit_and_publish(uow, self._bus)

            await self._publish_jobs(jobs)

    async def execute(self, job: Job, lease: Lease) -> None:
        """Run one claimed job to completion and advance the workflow.

        Every failure path ends with the job in a terminal or re-queued state:
        a job that is neither finished nor retryable is the one thing that would
        make a run hang forever.
        """
        try:
            await self._dispatch(job, lease)
        except Exception as exc:  # the retry policy decides what this means
            await self._handle_failure(job, lease, exc)
            return
        await self._advance(job.run_id)

    async def handle_dead_job(self, job: Job) -> None:
        """A job that exhausted its attempts must not leave its run waiting.

        Silence is the worst outcome here: the client would watch a run that can
        no longer progress. The run is failed with the job's own failure kind so
        the cause survives into the API response.
        """
        now = self._clock.now()
        async with self._coordinator.lock(job.run_id), self._uow_factory() as uow:
            run = await uow.runs.get(job.run_id)
            if run is None or run.is_terminal:
                return
            run.fail(
                now=now,
                kind=job.failure_kind or FailureKind.INFRASTRUCTURE,
                reason=job.failure_reason
                or f"{job.type} job exhausted its retry budget after {job.attempt} attempt(s)",
            )
            await uow.runs.update(run)
            uow.collect(run)
            await commit_and_publish(uow, self._bus)
        await self._workspaces.release_run(job.run_id)

    async def advance_stalled_runs(self) -> Sequence[RunId]:
        """Push along every run that has stopped without finishing.

        Two shapes, one cause: ``_advance`` runs exactly once, when a job ends.

        A run created over HTTP was never advanced at all — the API persists it
        and answers rather than holding a client on a GPU, and nothing did the
        second half. And a run whose last job finished could miss its one call
        to ``_advance`` — a restart, a crash, a lost race with the job's own
        status write — after which nothing would ever call it again. That one
        was seen on a real stack: every job SUCCEEDED, the candidate validated,
        the review PASS, and the run sat in REVIEWING with no error anywhere.

        Idempotent by construction: a run with a job still in flight is skipped,
        so a sweep can never hand the same candidate to a second worker.
        """
        async with self._uow_factory() as uow:
            stalled: list[RunId] = []
            for run in await uow.runs.list_active():
                if run.is_terminal or run.is_cancelling:
                    continue
                if run.status is RunStatus.CREATED:
                    stalled.append(run.id)
                    continue
                jobs = await uow.jobs.list_by_run(run.id)
                # No jobs at all is not "stalled": the run is between states
                # inside a transaction that has not committed yet.
                if jobs and not _has_unfinished(list(jobs)):
                    stalled.append(run.id)
        return await self._advance_each(stalled, what="advance")

    async def approve(self, run_id: RunId) -> Run:
        """Let the selected patch land. The write nobody could undo automatically."""
        async with self._coordinator.lock(run_id), self._uow_factory() as uow:
            run = await self._require_run(uow, run_id)
            if run.status is not RunStatus.AWAITING_APPROVAL:
                raise RunNotModifiableError(run_id, run.status)
            project = await self._require_project(uow, run)
            candidates = list(await uow.candidates.list_by_run(run_id))
            winner = next((c for c in candidates if c.id == run.selected_candidate_id), None)
            if winner is None:
                raise DomainError("the run has no selected candidate", run_id=str(run_id))

            now = self._clock.now()
            if self._config.integrate_on_success:
                await self._integrate(project=project, run=run, candidate=winner)
            for loser in candidates:
                if loser.id != winner.id:
                    loser.reject(now=now, reason="another candidate was approved")
                    await uow.candidates.update(loser)
            winner.select(now)
            await uow.candidates.update(winner)
            run.complete(now=now, candidate_id=winner.id)
            await self._workspaces.release_run(run.id)
            await uow.runs.update(run)
            uow.collect(run, *candidates)
            await commit_and_publish(uow, self._bus)
            return run

    async def reject(self, run_id: RunId, *, reason: str) -> Run:
        """Refuse the patch and send the reason back to the coder.

        A refusal is feedback, not a verdict: the reviewer passed it and a human
        did not, and the coder is the one who can act on the difference. When
        the repair budget is spent the run fails, carrying the human's reason
        rather than a generic one.
        """
        follow_ups: list[Job] = []
        async with self._coordinator.lock(run_id), self._uow_factory() as uow:
            run = await self._require_run(uow, run_id)
            if run.status is not RunStatus.AWAITING_APPROVAL:
                raise RunNotModifiableError(run_id, run.status)
            candidates = list(await uow.candidates.list_by_run(run_id))
            now = self._clock.now()
            run.reject_approval(now=now, reason=reason)

            target = [c for c in candidates if c.id == run.selected_candidate_id] or candidates
            follow_ups = await self._repair_or_fail(
                uow=uow, run=run, candidates=target, reason=f"rejected by a human: {reason}"
            )
            await self._persist_jobs(uow, follow_ups)
            await uow.runs.update(run)
            uow.collect(run, *candidates)
            await commit_and_publish(uow, self._bus)
            approved = run
        await self._publish_jobs(follow_ups)
        return approved

    async def resume_active_runs(self) -> Sequence[RunId]:
        """Re-schedule work for runs left in flight by an orchestrator restart.

        Durable state lives in PostgreSQL, so a restart loses nothing; what it
        does lose is the in-memory intent to act, which this restores.
        """
        async with self._uow_factory() as uow:
            runs = await uow.runs.list_active()
            run_ids = [run.id for run in runs]
        return await self._advance_each(run_ids, what="resume after a restart")

    async def _advance_each(self, run_ids: Sequence[RunId], *, what: str) -> Sequence[RunId]:
        """Advance many runs, letting no single one take down the caller.

        Both callers sweep every active run: one at startup, one on a timer.
        Without this isolation a single unrecoverable row — a workspace the
        restart destroyed, say — raised out of startup and put the whole
        control plane in a crash loop, serving nothing and advancing nothing.

        A run that cannot be advanced is failed rather than left active, or it
        would be retried on every boot forever while still reading as in-flight
        to anyone asking the API.
        """
        advanced: list[RunId] = []
        for run_id in run_ids:
            try:
                await self._advance(run_id)
            except Exception as exc:
                _log.exception("could not %s run %s; failing it", what, run_id)
                await self._fail_unrecoverable(run_id, what=what, exc=exc)
                continue
            advanced.append(run_id)
        return advanced

    async def _fail_unrecoverable(self, run_id: RunId, *, what: str, exc: Exception) -> None:
        """Record why a run can never continue. Best effort, and silent if even
        that fails: the caller is a startup path and must still come up."""
        with contextlib.suppress(Exception):
            async with self._coordinator.lock(run_id), self._uow_factory() as uow:
                run = await uow.runs.get(run_id)
                if run is None or run.is_terminal:
                    return
                run.fail(
                    now=self._clock.now(),
                    kind=_classify(exc),
                    reason=f"could not {what} this run: {exc}",
                )
                await uow.runs.update(run)
                uow.collect(run)
                await commit_and_publish(uow, self._bus)
        with contextlib.suppress(Exception):
            await self._workspaces.release_run(run_id)

    # ------------------------------------------------------------------
    # job dispatch
    # ------------------------------------------------------------------
    async def _dispatch(self, job: Job, lease: Lease) -> None:
        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            if run.is_cancelling or run.is_terminal:
                job.cancel(self._clock.now())
                await uow.jobs.update(job)
                await uow.commit()
                await self._queue.acknowledge(job_id=job.id, token=lease.token)
                return

        if job.type is JobType.PLAN:
            await self._run_planner(job, lease)
        elif job.type is JobType.CODE:
            await self._run_coder(job, lease)
        elif job.type is JobType.REVIEW:
            await self._run_reviewer(job, lease)
        elif job.type in _VALIDATION_STAGES:
            await self._run_validation_stage(job, lease)
        else:
            raise DomainError(f"no handler for job type {job.type}", job_type=str(job.type))

    # -- planner --------------------------------------------------------
    async def _run_planner(self, job: Job, lease: Lease) -> None:
        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            project = await self._require_project(uow, run)
            previous = await uow.plans.latest_for_run(run.id)

        workspace = await self._workspaces.create(
            project=project, run_id=run.id, role=WorkspaceRole.PLANNER
        )
        try:
            context = await self._build_context(
                workspace=workspace, run=run, role=AgentRole.PLANNER
            )
            requirements = JobRequirements(
                role=AgentRole.PLANNER,
                estimated_prompt_tokens=context.estimated_tokens,
                reserved_output_tokens=self._config.reserved_output_tokens,
            )
            async with self._pool.acquire(requirements) as acquired:
                outcome = await self._planner.plan(
                    provider=acquired.provider,
                    project=project,
                    run=run,
                    context=context,
                    previous_plan=previous,
                )
                worker_id = acquired.worker.id
        finally:
            await self._workspaces.release(workspace.id)

        now = self._clock.now()
        plan = self._plan_from_draft(
            draft=outcome.draft, run=run, prompt_version=outcome.prompt_version
        )

        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            await uow.plans.add(plan)
            run.plan_ready(plan_id=plan.id, task_count=plan.task_count, now=now)
            run.add_token_usage(outcome.usage, now)
            job.complete(
                token=lease.token,
                now=now,
                result={
                    "plan_id": str(plan.id),
                    "task_count": plan.task_count,
                    "worker_id": str(worker_id),
                    "prompt_version": outcome.prompt_version,
                    "repairs": outcome.repairs,
                },
            )
            await uow.jobs.update(job)
            await uow.runs.update(run)
            uow.collect(run, job)
            await commit_and_publish(uow, self._bus)
        await self._queue.acknowledge(job_id=job.id, token=lease.token)

    # -- coder ----------------------------------------------------------
    async def _run_coder(self, job: Job, lease: Lease) -> None:
        candidate_id = job.candidate_id
        if candidate_id is None:
            raise DomainError("a CODE job must name its candidate", job_id=str(job.id))

        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            project = await self._require_project(uow, run)
            candidate = await self._require_candidate(uow, candidate_id)
            plan = await uow.plans.latest_for_run(run.id)
            reviews = await uow.reviews.list_by_candidate(candidate_id)
            evidence = list(await uow.tool_results.list_by_candidate(candidate_id))

        workspace = await self._require_workspace(candidate)
        context = await self._build_context(
            workspace=workspace,
            run=run,
            role=AgentRole.CODER,
            candidate_id=candidate_id,
            # On a repair, the files this candidate already touched are put in
            # front of it rather than searched for. Whether a term from the
            # objective happens to match the file it wrote is luck, and a coder
            # asked to fix code it cannot see rewrites it from scratch.
            paths=_files_to_put_in_front_of_the_coder(candidate, plan),
        )
        repair_brief = accumulated_repair_brief(reviews)

        requirements = JobRequirements(
            role=AgentRole.CODER,
            estimated_prompt_tokens=context.estimated_tokens,
            requires_tools=True,
            reserved_output_tokens=self._config.reserved_output_tokens,
        )
        async with self._pool.acquire(requirements) as acquired:
            outcome = await self._coder.code(
                provider=acquired.provider,
                project=project,
                run=run,
                candidate=candidate,
                context=context,
                plan=plan,
                repair_brief=repair_brief,
                tool_output=_recent_tool_evidence(evidence),
            )
            worker_id = acquired.worker.id

        draft: CodeDraft = outcome.draft
        patch = await self._materialize_patch(workspace=workspace, draft=draft)

        now = self._clock.now()
        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            candidate = await self._require_candidate(uow, candidate_id)
            candidate.record_worker(worker_id)
            candidate.submit_patch(
                patch=patch,
                now=now,
                summary=draft.summary,
                uncertainties=draft.uncertainties,
            )
            candidate.start_validation(now)
            run.add_token_usage(outcome.usage, now)

            job.complete(
                token=lease.token,
                now=now,
                result={
                    "candidate_id": str(candidate_id),
                    "changed_files": len(patch.files),
                    "worker_id": str(worker_id),
                    "prompt_version": outcome.prompt_version,
                },
            )
            await uow.jobs.update(job)
            await uow.candidates.update(candidate)
            await uow.runs.update(run)
            uow.collect(run, job, candidate)
            await commit_and_publish(uow, self._bus)
        await self._queue.acknowledge(job_id=job.id, token=lease.token)

    # -- deterministic validation ---------------------------------------
    async def _run_validation_stage(self, job: Job, lease: Lease) -> None:
        candidate_id = job.candidate_id
        if candidate_id is None:
            raise DomainError("a validation job must name its candidate", job_id=str(job.id))

        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            project = await self._require_project(uow, run)
            candidate = await self._require_candidate(uow, candidate_id)

        workspace = await self._require_workspace(candidate)
        tool_name, kind = _STAGE_TOOLS[job.type]
        executor = self._tools.for_project(project, role=AgentRole.CODER)
        available = set(self._tools.available_tools(project, role=AgentRole.CODER))
        if tool_name not in available:
            # The project configures no such command. Skipping is honest; the
            # report records "skipped" rather than a fabricated pass.
            results: Sequence[ToolResult] = ()
        else:
            results = [
                await executor.execute(
                    invocation=ToolInvocation(
                        tool=tool_name,
                        kind=kind,
                        limits=ExecutionLimits(
                            timeout_seconds=self._config.validation_timeout_seconds
                        ),
                    ),
                    workspace=workspace,
                )
            ]

        now = self._clock.now()
        async with self._uow_factory() as uow:
            candidate = await self._require_candidate(uow, candidate_id)
            candidate.append_validation(results=list(results), now=now)
            if results:
                await uow.tool_results.add_many(
                    run_id=job.run_id, candidate_id=candidate_id, results=list(results)
                )
            job.complete(
                token=lease.token,
                now=now,
                result={
                    "stage": str(job.type),
                    "exit_codes": [r.exit_code for r in results],
                    "skipped": not results,
                },
            )
            await uow.jobs.update(job)
            await uow.candidates.update(candidate)
            uow.collect(job, candidate)
            await commit_and_publish(uow, self._bus)
        await self._queue.acknowledge(job_id=job.id, token=lease.token)

    # -- reviewer -------------------------------------------------------
    async def _run_reviewer(self, job: Job, lease: Lease) -> None:
        candidate_id = job.candidate_id
        if candidate_id is None:
            raise DomainError("a REVIEW job must name its candidate", job_id=str(job.id))

        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            project = await self._require_project(uow, run)
            candidate = await self._require_candidate(uow, candidate_id)
            plan = await uow.plans.latest_for_run(run.id)
            previous = await uow.reviews.list_by_candidate(candidate_id)

        requirements = JobRequirements(
            role=AgentRole.REVIEWER,
            reserved_output_tokens=self._config.reserved_output_tokens,
        )
        async with self._pool.acquire(requirements) as acquired:
            outcome = await self._reviewer.review(
                provider=acquired.provider,
                project=project,
                run=run,
                candidate=candidate,
                validation=candidate.validation,
                plan=plan,
                previous_reviews=previous,
            )
            worker_id = acquired.worker.id

        draft: ReviewDraft = outcome.draft
        now = self._clock.now()
        review = Review(
            id=self._ids.next_id(ReviewId),
            run_id=job.run_id,
            candidate_id=candidate_id,
            verdict=draft.verdict,
            iteration=len(previous) + 1,
            created_at=now,
            summary=draft.summary,
            findings=tuple(
                ReviewFinding(
                    summary=f.summary,
                    severity=f.severity,
                    file=f.file,
                    line=f.line,
                    repair_instruction=f.repair_instruction,
                )
                for f in draft.findings
            ),
            prompt_version=outcome.prompt_version,
        )

        async with self._uow_factory() as uow:
            run = await self._require_run(uow, job.run_id)
            candidate = await self._require_candidate(uow, candidate_id)
            await uow.reviews.add(review)
            candidate.record_review(draft.verdict)
            run.add_token_usage(outcome.usage, now)
            job.complete(
                token=lease.token,
                now=now,
                result={
                    "verdict": str(draft.verdict),
                    "findings": len(draft.findings),
                    "worker_id": str(worker_id),
                    "prompt_version": outcome.prompt_version,
                },
            )
            await uow.jobs.update(job)
            await uow.candidates.update(candidate)
            await uow.runs.update(run)
            uow.collect(run, job, candidate)
            await commit_and_publish(uow, self._bus)
        await self._queue.acknowledge(job_id=job.id, token=lease.token)

    # ------------------------------------------------------------------
    # workflow advancement
    # ------------------------------------------------------------------
    async def _advance(self, run_id: RunId) -> None:
        """Decide the next step. Serialized per run so it is decided exactly once."""
        async with self._coordinator.lock(run_id):
            follow_ups: list[Job] = []
            async with self._uow_factory() as uow:
                run = await uow.runs.get(run_id)
                if run is None or run.is_terminal or run.is_cancelling:
                    return
                project = await self._require_project(uow, run)
                candidates = list(await uow.candidates.list_by_run(run_id))
                jobs = list(await uow.jobs.list_by_run(run_id))
                now = self._clock.now()

                if _has_unfinished(jobs):
                    return  # something is still running; it will call us back

                if run.status is RunStatus.CREATED:
                    # A run created over HTTP used to sit here forever: nothing
                    # called start(), and this method ignored CREATED. The API
                    # could accept a run that never executed.
                    follow_ups = await self._open_first_stage(uow, run=run, project=project)
                elif run.status is RunStatus.PLAN_READY:
                    run.start_coding(now)
                    follow_ups = await self._open_candidates(uow, run=run, project=project)
                elif run.status in (RunStatus.CODING, RunStatus.VALIDATING, RunStatus.REPAIRING):
                    follow_ups = await self._advance_candidates(
                        uow=uow, run=run, project=project, candidates=candidates
                    )
                elif run.status is RunStatus.REVIEWING:
                    follow_ups = await self._advance_review(
                        uow=uow, run=run, project=project, candidates=candidates
                    )

                await self._persist_jobs(uow, follow_ups)
                await uow.runs.update(run)
                uow.collect(run, *candidates)
                await commit_and_publish(uow, self._bus)

            await self._publish_jobs(follow_ups)

    async def _advance_candidates(
        self,
        *,
        uow: UnitOfWork,
        run: Run,
        project: Project,
        candidates: Sequence[Candidate],
    ) -> list[Job]:
        """Push each candidate through its remaining validation stages."""
        now = self._clock.now()
        follow_ups: list[Job] = []

        for candidate in candidates:
            if candidate.status is not CandidateStatus.VALIDATING:
                continue
            stage = _next_stage(project, candidate.validation)
            if stage is not None:
                follow_ups.append(
                    self._new_job(run, stage, candidate_id=candidate.id, priority=Priority.HIGH)
                )
                continue
            candidate.record_validation(
                now=now,
                static_analysis_is_blocking=self._config.static_analysis_is_blocking,
            )
            await uow.candidates.update(candidate)

        if follow_ups:
            if run.status is not RunStatus.VALIDATING:
                run.start_validating(now)
            return follow_ups

        settled = [c for c in candidates if c.status is CandidateStatus.VALIDATED]
        if not settled:
            return []

        viable = self._selection.viable(settled)
        if viable:
            best = self._selection.select(viable).winner or viable[0]
            if run.status is not RunStatus.REVIEWING:
                run.start_reviewing(candidate_id=best.id, now=now)
            return [
                self._new_job(run, JobType.REVIEW, role=AgentRole.REVIEWER, candidate_id=best.id)
            ]

        return await self._repair_or_fail(
            uow=uow,
            run=run,
            candidates=settled,
            reason="no candidate survived deterministic validation",
        )

    async def _advance_review(
        self,
        *,
        uow: UnitOfWork,
        run: Run,
        project: Project,
        candidates: Sequence[Candidate],
    ) -> list[Job]:
        now = self._clock.now()
        reviewed = [c for c in candidates if c.review_verdict is not None]
        passed = [c for c in reviewed if c.review_verdict is ReviewVerdict.PASS]

        if passed:
            selection = self._selection.select(passed)
            winner = selection.winner or passed[0]
            if self._config.require_approval:
                # Stop before the write. Losing candidates are left alone: the
                # human may reject this one, and the others are the alternatives.
                run.select_candidate(candidate_id=winner.id, rationale=selection.rationale, now=now)
                run.await_approval(candidate_id=winner.id, now=now)
                return []
            if self._config.integrate_on_success:
                await self._integrate(project=project, run=run, candidate=winner)
            for loser in candidates:
                if loser.id != winner.id:
                    loser.reject(now=now, reason="another candidate was selected")
                    await uow.candidates.update(loser)
            winner.select(now)
            await uow.candidates.update(winner)
            run.select_candidate(candidate_id=winner.id, rationale=selection.rationale, now=now)
            run.complete(now=now, candidate_id=winner.id)
            await self._workspaces.release_run(run.id)
            return []

        failing = next((c for c in reviewed if c.review_verdict is ReviewVerdict.FAIL), None)
        reason = "the reviewer rejected every candidate"
        return await self._repair_or_fail(
            uow=uow,
            run=run,
            candidates=[failing] if failing else list(candidates),
            reason=reason,
        )

    async def _repair_or_fail(
        self,
        *,
        uow: UnitOfWork,
        run: Run,
        candidates: Sequence[Candidate],
        reason: str,
    ) -> list[Job]:
        """Send the best candidate back to the coder, or end the run honestly."""
        target = candidates[0] if candidates else None
        timestamp = self._clock.now()
        if target is None or not run.request_repair(
            candidate_id=target.id, reason=reason, now=timestamp
        ):
            run.fail(
                now=timestamp,
                kind=FailureKind.REVIEW,
                reason=f"{reason}; repair budget exhausted",
            )
            await self._workspaces.release_run(run.id)
            return []

        target.start_repair(timestamp)
        await uow.candidates.update(target)
        run.start_coding(timestamp)
        return [
            self._new_job(
                run,
                JobType.CODE,
                role=AgentRole.CODER,
                candidate_id=target.id,
                priority=Priority.HIGH,
            )
        ]

    # ------------------------------------------------------------------
    # failure handling
    # ------------------------------------------------------------------
    async def _handle_failure(self, job: Job, lease: Lease, exc: Exception) -> None:
        kind = _classify(exc)
        decision = self._retry.decide(kind=kind, attempt=job.attempt)
        _log.warning("job %s (%s) failed: %s -> %s", job.id, job.type, exc, decision.action)

        now = self._clock.now()
        async with self._uow_factory() as uow:
            run = await uow.runs.get(job.run_id)
            retryable = decision.should_retry_job
            job.fail(token=None, now=now, kind=kind, reason=str(exc), retryable=retryable)
            await uow.jobs.update(job)

            if not retryable and run is not None and not run.is_terminal:
                if kind.is_code_defect:
                    # The produced code is wrong; that is the repair loop's
                    # business, and _advance will pick it up.
                    pass
                else:
                    run.fail(now=now, kind=kind, reason=str(exc))
                    await uow.runs.update(run)
                    uow.collect(run)
            uow.collect(job)
            await commit_and_publish(uow, self._bus)

        # The policy computes a delay and, until now, nothing honoured it. With
        # a single-worker fleet — which scale-to-zero makes ordinary — an
        # immediate requeue means the next attempt asks the same empty pool the
        # same question: a real run spent all three attempts in four seconds.
        await self._queue.release(
            job_id=job.id,
            token=lease.token,
            requeue=decision.should_retry_job,
            not_before=(now + decision.delay) if decision.delay else None,
        )
        if decision.should_retry_job:
            async with self._uow_factory() as uow:
                stored = await uow.jobs.get(job.id)
                if stored is not None:
                    stored.requeue(now=self._clock.now(), reason=decision.reason)
                    await uow.jobs.update(stored)
                    uow.collect(stored)
                    await commit_and_publish(uow, self._bus)
                    await self._queue.enqueue(stored)
        else:
            await self._advance(job.run_id)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    async def _open_first_stage(self, uow: UnitOfWork, *, run: Run, project: Project) -> list[Job]:
        """The first jobs of a run: plan, or code directly.

        Whether the planner is involved was decided when the run was created, by
        an explicit complexity policy; the orchestrator only obeys it.
        """
        now = self._clock.now()
        if bool(run.metadata.get("requires_plan", True)):
            run.start_planning(now)
            return [self._new_job(run, JobType.PLAN, role=AgentRole.PLANNER)]
        run.start_coding(now)
        return await self._open_candidates(uow, run=run, project=project)

    async def _open_candidates(self, uow: UnitOfWork, *, run: Run, project: Project) -> list[Job]:
        """Create the candidates and their isolated workspaces, then their jobs.

        Fan-out is capped by the run's candidate count; if a single worker is
        available the jobs simply queue behind each other, so correctness never
        depends on parallelism being available.
        """
        jobs: list[Job] = []
        now = self._clock.now()
        for index in range(run.candidate_count):
            candidate = Candidate.create(
                candidate_id=self._ids.next_id(CandidateId),
                run_id=run.id,
                index=index,
                now=now,
            )
            workspace = await self._workspaces.create(
                project=project,
                run_id=run.id,
                role=WorkspaceRole.CANDIDATE,
                candidate_id=candidate.id,
            )
            candidate.attach_workspace(workspace.id)
            candidate.start_coding(now=now)
            await uow.candidates.add(candidate)
            uow.collect(candidate)
            jobs.append(
                self._new_job(run, JobType.CODE, role=AgentRole.CODER, candidate_id=candidate.id)
            )
        return jobs

    async def _integrate(self, *, project: Project, run: Run, candidate: Candidate) -> None:
        workspace = await self._require_workspace(candidate)
        await self._workspaces.integrate(
            project=project,
            handle=workspace,
            message=f"agentic run {run.id}: {run.objective}",
        )

    async def _materialize_patch(self, *, workspace: WorkspaceHandle, draft: CodeDraft) -> Patch:
        """Turn the coder's answer into a patch that actually exists on disk.

        Whole files are preferred over a diff: writing them and asking git for
        the difference cannot fail on malformed hunk headers, which is how a
        coder job died twice in a row with "No valid patches in input" the first
        time this ran against a real model.

        Either way the patch is read back from the workspace, never taken from
        the model's word for it.
        """
        if draft.files:
            await self._workspaces.write_files(workspace, {f.path: f.content for f in draft.files})
        elif draft.diff.strip():
            await self._workspaces.apply_patch(workspace, Patch.from_unified_diff(draft.diff))
        return await self._workspaces.diff(workspace)

    def _plan_from_draft(self, *, draft: PlanDraft, run: Run, prompt_version: str) -> Plan:
        return Plan.create(
            plan_id=self._ids.next_id(PlanId),
            run_id=run.id,
            revision=run.plan_revisions,
            objective=draft.objective or run.objective,
            tasks=[
                PlanTask(
                    id=self._ids.next_id(TaskId),
                    key=task.key,
                    title=task.title,
                    description=task.description,
                    target_paths=task.target_paths,
                    depends_on=task.depends_on,
                    validation=task.validation,
                )
                for task in draft.tasks
            ],
            now=self._clock.now(),
            assumptions=draft.assumptions,
            constraints=draft.constraints,
            risk_areas=draft.risk_areas,
            validation_requirements=draft.validation_requirements,
            prompt_version=prompt_version,
        )

    def _new_job(
        self,
        run: Run,
        job_type: JobType,
        *,
        role: AgentRole | None = None,
        candidate_id: CandidateId | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> Job:
        return Job.create(
            job_id=self._ids.next_id(JobId),
            run_id=run.id,
            project_id=run.project_id,
            job_type=job_type,
            now=self._clock.now(),
            role=role,
            candidate_id=candidate_id,
            priority=priority,
            max_attempts=self._config.job_max_attempts,
        )

    async def _persist_jobs(self, uow: UnitOfWork, jobs: Sequence[Job]) -> None:
        """Store jobs as QUEUED before publishing them.

        Durable first, transport second: a crash in between leaves jobs the
        resume loop can re-publish, whereas the reverse order would hand a
        worker a job nothing remembers.
        """
        now = self._clock.now()
        for job in jobs:
            job.enqueue(now)
            await uow.jobs.add(job)
            uow.collect(job)

    async def _publish_jobs(self, jobs: Sequence[Job]) -> None:
        for job in jobs:
            await self._queue.enqueue(job)

    async def _build_context(
        self,
        *,
        workspace: WorkspaceHandle,
        run: Run,
        role: AgentRole,
        candidate_id: object = None,
        paths: Sequence[str] = (),
    ) -> RepositoryContext:
        """Build the view of the repository, and record what it contained.

        The manifest is published rather than kept: it is evidence about one
        inference, it reaches the live run stream so a viewer sees the
        selection while the run is still going, and it lands in the audit log
        for afterwards. An empty selection — the defect that had every agent
        inventing code from a filename list — shows up here as no files at all.
        """
        request = await self._context_request(run.objective, role=role, paths=paths)
        context = await self._context.build(workspace=workspace, request=request)
        event = RepositoryContextSelected(
            occurred_at=self._clock.now(),
            run_id=run.id,
            role=role,
            candidate_id=candidate_id,  # type: ignore[arg-type]
            files={e.path: e.estimated_tokens for e in context.excerpts},
            tree=tuple(context.file_tree),
            notes=tuple(context.notes),
            estimated_tokens=context.estimated_tokens,
            budget_tokens=request.max_tokens,
        )
        # Appended *and* published. Publishing alone reaches whoever is
        # watching right now and nothing else: the durable log is written by
        # the unit of work, so a live-only event vanishes from the replay and
        # a viewer that opens the run afterwards sees no manifest at all.
        async with self._uow_factory() as uow:
            await uow.events.append([event])
            await uow.commit()
        await self._bus.publish([event])
        return context

    async def _context_request(
        self, objective: str, *, role: AgentRole, paths: Sequence[str] = ()
    ) -> ContextRequest:
        """Size the repository view against what the fleet can actually hold.

        The configured ceiling was 24000 while the engines were served with
        MAX_MODEL_LEN=16384: the orchestrator asked for a view no worker could
        take, and the scheduler — which does check — would have refused the job
        the moment the context stopped being empty. Two numbers that had to
        agree, in two packages that never spoke.

        The fleet only ever lowers the ceiling. An empty fleet leaves it alone:
        there is nothing to learn from, and guessing is what caused this.
        """
        budget = await self._pool.prompt_budget(
            JobRequirements(role=role, reserved_output_tokens=self._config.reserved_output_tokens)
        )
        max_tokens = self._config.excerpt_budget(budget)
        if max_tokens != self._config.context_max_tokens:
            _log.debug(
                "context budget lowered from %d to %d by the fleet's context window",
                self._config.context_max_tokens,
                max_tokens,
            )
        return ContextRequest(
            objective=objective,
            paths=tuple(paths),
            max_files=self._config.context_max_files,
            max_tokens=max_tokens,
        )

    async def _require_workspace(self, candidate: Candidate) -> WorkspaceHandle:
        if candidate.workspace_id is None:
            raise WorkspaceError("candidate has no workspace", candidate_id=str(candidate.id))
        handle = await self._workspaces.get(candidate.workspace_id)
        if handle is None:
            raise WorkspaceError(
                "workspace no longer exists", workspace_id=str(candidate.workspace_id)
            )
        return handle

    @staticmethod
    async def _require_run(uow: UnitOfWork, run_id: RunId) -> Run:
        run = await uow.runs.get(run_id)
        if run is None:
            raise EntityNotFoundError("Run", run_id)
        return run

    @staticmethod
    async def _require_project(uow: UnitOfWork, run: Run) -> Project:
        project = await uow.projects.get(run.project_id)
        if project is None:
            raise EntityNotFoundError("Project", run.project_id)
        return project

    @staticmethod
    async def _require_candidate(uow: UnitOfWork, candidate_id: CandidateId) -> Candidate:
        candidate = await uow.candidates.get(candidate_id)
        if candidate is None:
            raise EntityNotFoundError("Candidate", candidate_id)
        return candidate


def _has_unfinished(jobs: Sequence[Job]) -> bool:
    return any(not job.status.is_terminal for job in jobs)


def _stage_is_configured(project: Project, stage: JobType) -> bool:
    toolchain = project.toolchain
    return bool(
        {
            JobType.BUILD: toolchain.build_command,
            JobType.TEST: toolchain.test_command,
            JobType.STATIC_ANALYSIS: toolchain.static_analysis_command,
        }.get(stage)
    )


def _next_stage(project: Project, report: ValidationReport) -> JobType | None:
    """The next configured stage that has not run yet.

    Stops at the first observed failure: running a test suite against a build
    that did not compile tells us nothing we do not already know.
    """
    done = {
        JobType.BUILD: report.build_passed,
        JobType.TEST: report.tests_passed,
        JobType.STATIC_ANALYSIS: report.static_analysis_passed,
    }
    for stage in _VALIDATION_STAGES:
        if done[stage] is False:
            return None
        if done[stage] is None and _stage_is_configured(project, stage):
            return stage
    return None


# Order matters: the first match wins, so a subclass must come before its base.
# OutputTruncatedError is a StructuredOutputError and must not be classified as
# one — re-asking a truncated answer reproduces the truncation exactly.
_FAILURE_KINDS: Final[tuple[tuple[type[Exception], FailureKind], ...]] = (
    (OutputTruncatedError, FailureKind.OUTPUT_TRUNCATED),
    (StructuredOutputError, FailureKind.INVALID_STRUCTURED_OUTPUT),
    (LLMTimeoutError, FailureKind.INFERENCE),
    (InferenceError, FailureKind.INFERENCE),
    (NoCompatibleWorkerError, FailureKind.NO_WORKER),
    (ToolExecutionError, FailureKind.TOOL),
    (WorkspaceError, FailureKind.INFRASTRUCTURE),
)


def _files_to_put_in_front_of_the_coder(
    candidate: Candidate, plan: Plan | None, *, limit: int = 12
) -> tuple[str, ...]:
    """Files this coder should not have to go looking for.

    Its own changes first, because a repair it cannot see turns into a rewrite.
    Then whatever the planner named: each task carries ``target_paths``, which
    is the planner saying plainly where the work belongs, and nothing read
    them — the first attempt searched for words from the objective and found
    the right file only when one happened to match.

    Bounded, because these are added on top of the search rather than instead
    of it, and the token budget is shared.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for path in candidate.patch.changed_paths if candidate.patch else ():
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    for task in plan.tasks if plan is not None else ():
        for path in task.target_paths:
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return tuple(ordered[:limit])


def accumulated_repair_brief(reviews: Sequence[Review]) -> str | None:
    """Every objection still on the table, with how often it was raised.

    The coder used to receive only the newest review while the reviewer
    received the whole history. A real run had the same objection raised in
    four consecutive rounds, worded identically each time, with nothing to
    tell the coder it was the fourth — so it kept changing other things.

    ``None`` when no review exists, which is a different thing from an empty
    brief and is what the prompt distinguishes.
    """
    if not reviews:
        return None
    counts: dict[str, int] = {}
    order: list[str] = []
    for review in reviews:
        for line in review.repair_brief().splitlines():
            if not line.strip():
                continue
            if line not in counts:
                order.append(line)
            counts[line] = counts.get(line, 0) + 1
    if not order:
        return None
    return "\n".join(
        line if counts[line] == 1 else f"{line}  [raised {counts[line]} times; still not fixed]"
        for line in order
    )


def _recent_tool_evidence(results: Sequence[ToolResult], *, limit: int = 3) -> str:
    """What the deterministic tools said, for the agent that can act on it.

    The reviewer has always received this and cannot change a line; the coder
    never did and is the only thing that can. Three real runs spent their whole
    repair budget re-deriving the same code because nothing told them what
    failed.

    Failures first and most recent first: a build that broke before the tests
    ran explains more than a test that never got the chance. An empty string
    when nothing has run yet, because inventing evidence is worse than none.
    """
    failures = [r for r in reversed(results) if not r.succeeded]
    chosen = failures[:limit] or list(reversed(results))[:1]
    return "\n\n".join(f"$ {r.command}\nexit={r.exit_code}\n{r.tail(2000)}" for r in chosen)


def _classify(exc: Exception) -> FailureKind:
    """Map an exception to the failure kind the retry policy branches on.

    Anything unrecognised is infrastructure: the safe assumption is that the
    machine failed, which is retried on another worker, rather than that the
    model did, which would send an unrelated error into the repair loop.
    """
    for error_type, kind in _FAILURE_KINDS:
        if isinstance(exc, error_type):
            return kind
    return FailureKind.INFRASTRUCTURE


def summarize_findings(draft: ReviewDraft) -> str:
    return format_findings(draft.findings)
