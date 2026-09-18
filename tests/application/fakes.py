"""In-memory doubles for the application tests.

They implement the domain ports faithfully enough that the orchestrator cannot
tell them from the real thing: leases expire, events are only published after a
commit, and the fake model answers with real structured output. Anything less
would let the workflow tests pass for the wrong reasons.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from uuid import UUID

from application.dto.agent_io import CodeDraft, FindingDraft, PlanDraft, ReviewDraft, TaskDraft
from application.ports import RenderedPrompt
from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan
from domain.entities.project import Project
from domain.entities.review import Review, Severity
from domain.entities.run import Run
from domain.entities.worker import Worker
from domain.enums import AgentRole, JobStatus, JobType, ReviewVerdict, RunStatus
from domain.events.base import DomainEvent
from domain.exceptions import JobLeaseExpiredError, LLMTimeoutError, StructuredOutputError
from domain.ports.repository_context import ContextRequest, FileExcerpt, RepositoryContext
from domain.value_objects.identifiers import (
    CandidateId,
    EntityId,
    IdempotencyKey,
    JobId,
    PlanId,
    ProjectId,
    RunId,
    WorkerId,
    WorkspaceId,
)
from domain.value_objects.lease import Lease, LeaseToken
from domain.value_objects.llm import (
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    FinishReason,
    ModelInfo,
    TokenUsage,
)
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolInvocation, ToolKind, ToolResult
from domain.value_objects.worker import WorkerLoad
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceKind, WorkspaceRole

__all__ = [
    "FakeClock",
    "FakeContextProvider",
    "FakeEventBus",
    "FakeIdGenerator",
    "FakeJobQueue",
    "FakeLLMProvider",
    "FakeLLMProviderFactory",
    "FakeOutputCodec",
    "FakePromptRenderer",
    "FakeToolExecutor",
    "FakeWorkerRegistry",
    "FakeWorkspaceManager",
    "InMemoryUnitOfWork",
    "uow_factory_for",
]


# ----------------------------------------------------------------------
# time and identity
# ----------------------------------------------------------------------
class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        return self._now


class FakeIdGenerator:
    """Sequential UUIDs, so failures name a readable identifier."""

    def __init__(self) -> None:
        self._counter = 0

    def next_id[T: EntityId](self, kind: type[T]) -> T:
        self._counter += 1
        return kind(UUID(int=self._counter))


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------
class _Repo[TId, TEntity]:
    def __init__(self) -> None:
        self.items: dict[TId, TEntity] = {}

    async def add(self, entity: Any) -> None:
        self.items[entity.id] = entity

    async def get(self, key: TId) -> TEntity | None:
        return self.items.get(key)

    async def update(self, entity: Any) -> None:
        self.items[entity.id] = entity


class _ProjectRepo(_Repo[ProjectId, Project]):
    async def get_by_name(self, name: str) -> Project | None:
        return next((p for p in self.items.values() if p.name == name), None)

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> Sequence[Project]:
        return list(self.items.values())[offset : offset + limit]


class _RunRepo(_Repo[RunId, Run]):
    async def find_by_idempotency_key(self, key: IdempotencyKey) -> Run | None:
        return next((r for r in self.items.values() if r.idempotency_key == key), None)

    async def list_by_project(
        self, project_id: ProjectId, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Run]:
        matches = [r for r in self.items.values() if r.project_id == project_id]
        return matches[offset : offset + limit]

    async def list_active(self) -> Sequence[Run]:
        return [r for r in self.items.values() if not r.status.is_terminal]

    async def count_by_status(self) -> dict[RunStatus, int]:
        counts: dict[RunStatus, int] = {}
        for run in self.items.values():
            counts[run.status] = counts.get(run.status, 0) + 1
        return counts


class _JobRepo(_Repo[JobId, Job]):
    async def find_by_idempotency_key(self, key: IdempotencyKey) -> Job | None:
        return next((j for j in self.items.values() if j.idempotency_key == key), None)

    async def list_by_run(self, run_id: RunId) -> Sequence[Job]:
        return [j for j in self.items.values() if j.run_id == run_id]

    async def list_by_status(self, status: JobStatus, *, limit: int = 100) -> Sequence[Job]:
        return [j for j in self.items.values() if j.status is status][:limit]

    async def list_expired_leases(self, *, now: datetime, limit: int = 100) -> Sequence[Job]:
        return [j for j in self.items.values() if j.lease is not None and j.lease.is_expired(now)][
            :limit
        ]


class _PlanRepo(_Repo[PlanId, Plan]):
    async def latest_for_run(self, run_id: RunId) -> Plan | None:
        plans = [p for p in self.items.values() if p.run_id == run_id]
        return max(plans, key=lambda p: p.revision, default=None)

    async def list_by_run(self, run_id: RunId) -> Sequence[Plan]:
        return sorted(
            (p for p in self.items.values() if p.run_id == run_id), key=lambda p: p.revision
        )


class _CandidateRepo(_Repo[CandidateId, Candidate]):
    async def list_by_run(self, run_id: RunId) -> Sequence[Candidate]:
        return sorted((c for c in self.items.values() if c.run_id == run_id), key=lambda c: c.index)


class _ReviewRepo:
    def __init__(self) -> None:
        self.items: list[Review] = []

    async def add(self, review: Review) -> None:
        self.items.append(review)

    async def list_by_candidate(self, candidate_id: CandidateId) -> Sequence[Review]:
        return [r for r in self.items if r.candidate_id == candidate_id]

    async def latest_for_candidate(self, candidate_id: CandidateId) -> Review | None:
        matches = await self.list_by_candidate(candidate_id)
        return matches[-1] if matches else None


class _ToolResultRepo:
    def __init__(self) -> None:
        self.items: list[tuple[RunId, CandidateId | None, ToolResult]] = []

    async def add_many(
        self, *, run_id: RunId, candidate_id: CandidateId | None, results: Sequence[ToolResult]
    ) -> None:
        self.items.extend((run_id, candidate_id, r) for r in results)

    async def list_by_candidate(self, candidate_id: CandidateId) -> Sequence[ToolResult]:
        return [r for _, cid, r in self.items if cid == candidate_id]


class _EventStore:
    def __init__(self) -> None:
        self.items: list[tuple[int, DomainEvent]] = []

    async def append(self, events: Sequence[DomainEvent]) -> None:
        for event in events:
            self.items.append((len(self.items) + 1, event))

    async def list_by_run(
        self, run_id: RunId, *, after_sequence: int | None = None, limit: int = 500
    ) -> Sequence[tuple[int, DomainEvent]]:
        floor = after_sequence or 0
        matches = [
            (seq, ev)
            for seq, ev in self.items
            if seq > floor and str(getattr(ev, "run_id", "")) == str(run_id)
        ]
        return matches[:limit]


class _Store:
    """The shared state behind every unit of work, as a database would be."""

    def __init__(self) -> None:
        self.projects = _ProjectRepo()
        self.runs = _RunRepo()
        self.jobs = _JobRepo()
        self.plans = _PlanRepo()
        self.candidates = _CandidateRepo()
        self.reviews = _ReviewRepo()
        self.tool_results = _ToolResultRepo()
        self.events = _EventStore()


class InMemoryUnitOfWork:
    """Drains aggregate events into the store on commit, never before."""

    def __init__(self, store: _Store) -> None:
        self._store = store
        self.projects = store.projects
        self.runs = store.runs
        self.jobs = store.jobs
        self.plans = store.plans
        self.candidates = store.candidates
        self.reviews = store.reviews
        self.tool_results = store.tool_results
        self.events = store.events
        self._collected: list[Any] = []
        self._drained: list[DomainEvent] = []
        self.committed = False

    async def __aenter__(self) -> InMemoryUnitOfWork:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if not self.committed:
            await self.rollback()

    def collect(self, *entities: object) -> None:
        self._collected.extend(entities)

    @property
    def collected_events(self) -> Sequence[DomainEvent]:
        return tuple(self._drained)

    async def commit(self) -> None:
        drained: list[DomainEvent] = []
        for entity in self._collected:
            pull = getattr(entity, "pull_events", None)
            if pull is not None:
                drained.extend(pull())
        await self._store.events.append(drained)
        self._drained = drained
        self._collected.clear()
        self.committed = True

    async def rollback(self) -> None:
        # Buffered events die with the transaction: nothing was published.
        for entity in self._collected:
            pull = getattr(entity, "pull_events", None)
            if pull is not None:
                pull()
        self._collected.clear()


def uow_factory_for(store: _Store):
    def factory() -> InMemoryUnitOfWork:
        return InMemoryUnitOfWork(store)

    return factory


# ----------------------------------------------------------------------
# messaging
# ----------------------------------------------------------------------
class FakeEventBus:
    def __init__(self) -> None:
        self.published: list[DomainEvent] = []
        self._subscribers: list[asyncio.Queue[DomainEvent]] = []

    async def publish(self, events: Sequence[DomainEvent]) -> None:
        self.published.extend(events)
        for queue in self._subscribers:
            for event in events:
                queue.put_nowait(event)

    async def subscribe(self, run_id: RunId) -> AsyncIterator[DomainEvent]:
        queue: asyncio.Queue[DomainEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                if str(getattr(event, "run_id", "")) == str(run_id):
                    yield event
        finally:
            self._subscribers.remove(queue)

    def names(self) -> list[str]:
        return [e.name for e in self.published]


class FakeJobQueue:
    """Priority queue with real leases, so lease expiry is testable."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self._queued: list[Job] = []
        self._leased: dict[JobId, tuple[Job, Lease]] = {}
        self.enqueued: list[JobId] = []

    async def enqueue(self, job: Job) -> None:
        if any(j.id == job.id for j in self._queued) or job.id in self._leased:
            return  # at-least-once delivery must not duplicate work
        self._queued.append(job)
        self.enqueued.append(job.id)

    async def claim(
        self,
        *,
        consumer: WorkerId,
        job_types: Sequence[JobType],
        lease_duration: timedelta,
        now: datetime,
    ) -> tuple[Job, Lease] | None:
        eligible = [j for j in self._queued if j.type in job_types]
        if not eligible:
            return None
        job = max(eligible, key=lambda j: (j.priority.rank, -self._queued.index(j)))
        self._queued.remove(job)
        lease = job.lease_to(worker_id=consumer, now=now, duration=lease_duration)
        self._leased[job.id] = (job, lease)
        return job, lease

    async def renew(
        self, *, job_id: JobId, token: LeaseToken, duration: timedelta, now: datetime
    ) -> Lease:
        entry = self._leased.get(job_id)
        if entry is None or entry[1].token != token or entry[1].is_expired(now):
            raise JobLeaseExpiredError(job_id)
        job, _ = entry
        lease = job.renew_lease(token=token, now=now, duration=duration)
        self._leased[job_id] = (job, lease)
        return lease

    async def acknowledge(self, *, job_id: JobId, token: LeaseToken) -> None:
        self._leased.pop(job_id, None)

    async def release(self, *, job_id: JobId, token: LeaseToken, requeue: bool) -> None:
        entry = self._leased.pop(job_id, None)
        if entry is not None and requeue:
            self._queued.append(entry[0])

    async def reclaim_expired(self, *, now: datetime, limit: int = 100) -> Sequence[JobId]:
        expired = [jid for jid, (_, lease) in self._leased.items() if lease.is_expired(now)]
        for job_id in expired[:limit]:
            self._leased.pop(job_id, None)
        return expired[:limit]

    async def cancel_run_jobs(self, run_id: RunId) -> Sequence[JobId]:
        dropped = [j.id for j in self._queued if j.run_id == run_id]
        self._queued = [j for j in self._queued if j.run_id != run_id]
        return dropped

    async def depth(self, job_type: JobType | None = None) -> int:
        if job_type is None:
            return len(self._queued)
        return sum(1 for j in self._queued if j.type is job_type)

    @property
    def pending(self) -> int:
        return len(self._queued)


class FakeWorkerRegistry:
    def __init__(self) -> None:
        self.workers: dict[WorkerId, Worker] = {}

    async def register(self, worker: Worker, *, ttl: timedelta) -> None:
        self.workers[worker.id] = worker

    async def heartbeat(
        self, worker_id: WorkerId, *, load: WorkerLoad, at: datetime, ttl: timedelta
    ) -> bool:
        return worker_id in self.workers

    async def get(self, worker_id: WorkerId) -> Worker | None:
        return self.workers.get(worker_id)

    async def list_all(self) -> Sequence[Worker]:
        return list(self.workers.values())

    async def list_available(self) -> Sequence[Worker]:
        return [w for w in self.workers.values() if w.status.accepts_new_jobs]

    async def update(self, worker: Worker) -> None:
        self.workers[worker.id] = worker

    async def deregister(self, worker_id: WorkerId) -> None:
        self.workers.pop(worker_id, None)

    async def reap_stale(self, *, now: datetime, heartbeat_timeout: timedelta) -> Sequence[Worker]:
        reaped: list[Worker] = []
        for worker in list(self.workers.values()):
            if worker.is_stale(now, heartbeat_timeout) and worker.status.is_live:
                worker.mark_unavailable(now=now, reason="heartbeat timeout")
                reaped.append(worker)
        return reaped


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
class FakeLLMProvider:
    """Answers with real, schema-valid JSON for each role.

    Scriptable so a test can inject exactly the pathology it wants: a malformed
    answer, a timeout, a truncated completion. Anything the orchestrator must
    survive has to be reproducible here.
    """

    def __init__(
        self,
        *,
        model_id: str = "fake-coder",
        context_length: int = 32_000,
        fail_with: Exception | None = None,
        finish_reason: FinishReason = FinishReason.STOP,
    ) -> None:
        self._model_id = model_id
        self._context_length = context_length
        self._fail_with = fail_with
        self._finish_reason = finish_reason
        self._scripted: dict[AgentRole, list[str]] = {}
        self.calls: list[CompletionRequest] = []

    def script(self, role: AgentRole, *answers: str) -> None:
        """Queue answers for one role.

        Scripting per role rather than as one flat list matters: the order in
        which the orchestrator interleaves roles is its business, and a test
        that depends on it would break for reasons that are not bugs.
        """
        self._scripted.setdefault(role, []).extend(answers)

    @staticmethod
    def answer_for(role: AgentRole) -> str:
        """The default well-formed answer for a role."""
        return _DEFAULT_ANSWERS[role]

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(model_id=self._model_id, context_length=self._context_length)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        role = _role_of(request)
        queued = self._scripted.get(role)
        content = queued.pop(0) if queued else _DEFAULT_ANSWERS[role]
        return CompletionResult(
            content=content,
            model=self._model_id,
            finish_reason=self._finish_reason,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            latency_ms=5,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        result = await self.complete(request)
        yield result.content

    async def health(self) -> bool:
        return True


_DEFAULT_ANSWERS: dict[AgentRole, str] = {
    AgentRole.PLANNER: json.dumps(
        {
            "objective": "implement the objective",
            "tasks": [
                {"key": "impl", "title": "Implement", "depends_on": []},
                {"key": "test", "title": "Add tests", "depends_on": ["impl"]},
            ],
            "assumptions": ["the build command is configured"],
            "constraints": [],
            "risk_areas": [],
            "validation_requirements": ["tests must pass"],
        }
    ),
    AgentRole.CODER: json.dumps(
        {
            "summary": "added the parser",
            "diff": (
                "diff --git a/parser.py b/parser.py\n"
                "--- a/parser.py\n+++ b/parser.py\n"
                "@@ -0,0 +1,2 @@\n+def parse():\n+    return 42\n"
            ),
            "uncertainties": [],
            "done": True,
        }
    ),
    AgentRole.REVIEWER: json.dumps({"verdict": "PASS", "summary": "looks correct", "findings": []}),
}


def _role_of(request: CompletionRequest) -> AgentRole:
    """Read back the role marker the fake prompt renderer injects."""
    text = "\n".join(m.content for m in request.messages)
    for role in AgentRole:
        if f"ROLE:{role}" in text:
            return role
    raise AssertionError("the prompt carried no role marker")


class FakeLLMProviderFactory:
    def __init__(self, provider: FakeLLMProvider | None = None) -> None:
        self.provider = provider or FakeLLMProvider()
        self.endpoints: list[str] = []

    def for_endpoint(self, endpoint: Any, *, model_id: str) -> FakeLLMProvider:
        self.endpoints.append(str(endpoint))
        return self.provider


class FakePromptRenderer:
    """Renders a marker the fake provider keys on, plus the variables."""

    VERSION = "v1-test"

    def render(
        self, *, role: AgentRole, variables: Mapping[str, Any], version: str | None = None
    ) -> RenderedPrompt:
        body = "\n".join(f"{k}: {v}" for k, v in sorted(variables.items()))
        return RenderedPrompt(
            messages=(
                ChatMessage.system(f"ROLE:{role} respond with JSON only"),
                ChatMessage.user(body),
            ),
            version=version or self.VERSION,
            role=role,
        )

    def current_version(self, role: AgentRole) -> str:
        return self.VERSION


class FakeOutputCodec:
    """Parses the fake provider's JSON into drafts, rejecting anything else."""

    def schema_for(self, role: AgentRole) -> Mapping[str, Any]:
        return {"type": "object", "title": str(role)}

    def _load(self, raw: str, role: AgentRole) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"not valid JSON: {exc}", schema=str(role), raw_output=raw
            ) from exc
        if not isinstance(payload, dict):
            raise StructuredOutputError("expected a JSON object", schema=str(role), raw_output=raw)
        return payload

    def parse_plan(self, raw: str) -> PlanDraft:
        payload = self._load(raw, AgentRole.PLANNER)
        return PlanDraft(
            objective=str(payload.get("objective", "")),
            tasks=tuple(
                TaskDraft(
                    key=str(t["key"]),
                    title=str(t["title"]),
                    description=str(t.get("description", "")),
                    depends_on=tuple(t.get("depends_on", ())),
                )
                for t in payload.get("tasks", [])
            ),
            assumptions=tuple(payload.get("assumptions", ())),
            constraints=tuple(payload.get("constraints", ())),
            risk_areas=tuple(payload.get("risk_areas", ())),
            validation_requirements=tuple(payload.get("validation_requirements", ())),
        )

    def parse_code(self, raw: str) -> CodeDraft:
        payload = self._load(raw, AgentRole.CODER)
        return CodeDraft(
            summary=str(payload.get("summary", "")),
            diff=str(payload.get("diff", "")),
            uncertainties=tuple(payload.get("uncertainties", ())),
            done=bool(payload.get("done", True)),
        )

    def parse_review(self, raw: str) -> ReviewDraft:
        payload = self._load(raw, AgentRole.REVIEWER)
        return ReviewDraft(
            verdict=ReviewVerdict(payload["verdict"]),
            summary=str(payload.get("summary", "")),
            findings=tuple(
                FindingDraft(
                    summary=str(f["summary"]),
                    severity=Severity(f.get("severity", "MAJOR")),
                    file=f.get("file"),
                    line=f.get("line"),
                    repair_instruction=f.get("repair_instruction"),
                )
                for f in payload.get("findings", [])
            ),
        )

    def repair_messages(self, *, role: AgentRole, raw: str, error: str) -> Sequence[ChatMessage]:
        return (
            ChatMessage.assistant(raw),
            ChatMessage.user(f"That answer was rejected: {error}. Answer again, JSON only."),
        )


# ----------------------------------------------------------------------
# workspaces, tools and context
# ----------------------------------------------------------------------
class FakeWorkspaceManager:
    """Tracks handles and patches in memory, enforcing path uniqueness."""

    def __init__(self, ids: FakeIdGenerator) -> None:
        self._ids = ids
        self.handles: dict[WorkspaceId, WorkspaceHandle] = {}
        self.patches: dict[WorkspaceId, Patch] = {}
        self.integrated: list[WorkspaceId] = []
        self.released: list[WorkspaceId] = []

    async def create(
        self,
        *,
        project: Project,
        run_id: RunId,
        role: WorkspaceRole,
        candidate_id: CandidateId | None = None,
        base_revision: str | None = None,
    ) -> WorkspaceHandle:
        workspace_id = self._ids.next_id(WorkspaceId)
        path = f"/workspaces/{run_id}/{role}-{workspace_id}"
        if any(h.path == path for h in self.handles.values()):
            raise AssertionError(f"two workspaces would share the path {path}")
        handle = WorkspaceHandle(
            id=workspace_id,
            run_id=run_id,
            role=role,
            kind=WorkspaceKind.GIT_WORKTREE,
            path=path,
            base_revision=base_revision,
            candidate_id=candidate_id,
        )
        self.handles[workspace_id] = handle
        return handle

    async def get(self, workspace_id: WorkspaceId) -> WorkspaceHandle | None:
        return self.handles.get(workspace_id)

    async def diff(self, handle: WorkspaceHandle) -> Patch:
        return self.patches.get(handle.id, Patch(diff=""))

    async def apply_patch(self, handle: WorkspaceHandle, patch: Patch) -> None:
        if not handle.is_writable:
            raise AssertionError(f"{handle.role} workspaces are read-only")
        self.patches[handle.id] = patch

    async def commit(self, handle: WorkspaceHandle, *, message: str) -> str:
        return f"rev-{handle.id}"

    async def integrate(self, *, project: Project, handle: WorkspaceHandle, message: str) -> str:
        self.integrated.append(handle.id)
        return f"merged-{handle.id}"

    async def release(self, workspace_id: WorkspaceId) -> None:
        self.handles.pop(workspace_id, None)
        self.released.append(workspace_id)

    async def release_run(self, run_id: RunId) -> Sequence[WorkspaceId]:
        doomed = [wid for wid, h in self.handles.items() if h.run_id == run_id]
        for workspace_id in doomed:
            await self.release(workspace_id)
        return doomed


class FakeToolExecutor:
    """Returns scripted exit codes per tool, defaulting to success.

    Doubles as its own factory: which tools exist depends on the project's
    toolchain, exactly as the real registry does, so a project with no test
    command really has no ``run_tests`` tool to call.
    """

    STAGE_TOOLS: ClassVar[Mapping[str, str]] = {
        "build": "build_command",
        "run_tests": "test_command",
        "static_analysis": "static_analysis_command",
    }

    def __init__(self, exit_codes: Mapping[str, int] | None = None) -> None:
        self.exit_codes = dict(exit_codes or {})
        self.invocations: list[ToolInvocation] = []

    def for_project(self, project: Project, *, role: AgentRole) -> FakeToolExecutor:
        return self

    def available_tools(self, project: Project, *, role: AgentRole) -> Sequence[str]:
        return tuple(
            name
            for name, attribute in self.STAGE_TOOLS.items()
            if getattr(project.toolchain, attribute)
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        self.invocations.append(invocation)
        command = str(invocation.arguments.get("command", invocation.tool))
        exit_code = self.exit_codes.get(invocation.tool, 0)
        return ToolResult(
            tool=invocation.tool,
            kind=invocation.kind,
            command=command,
            exit_code=exit_code,
            stdout="ok" if exit_code == 0 else "",
            stderr="" if exit_code == 0 else "failure",
            duration_ms=10,
        )

    async def execute_many(
        self, *, invocations: Sequence[ToolInvocation], workspace: WorkspaceHandle
    ) -> Sequence[ToolResult]:
        results = []
        for invocation in invocations:
            result = await self.execute(invocation=invocation, workspace=workspace)
            results.append(result)
            if not result.succeeded:
                break
        return results


class FakeContextProvider:
    def __init__(self, estimated_tokens: int = 1000) -> None:
        self._estimated = estimated_tokens

    async def build(
        self, *, workspace: WorkspaceHandle, request: ContextRequest
    ) -> RepositoryContext:
        return RepositoryContext(
            excerpts=(FileExcerpt(path="parser.py", content="# existing code"),),
            file_tree=("parser.py", "tests/test_parser.py"),
            estimated_tokens=self._estimated,
        )

    async def search(
        self, *, workspace: WorkspaceHandle, pattern: str, limit: int = 100
    ) -> Sequence[FileExcerpt]:
        return ()

    async def read_file(
        self, *, workspace: WorkspaceHandle, path: str, max_bytes: int = 200_000
    ) -> FileExcerpt | None:
        return FileExcerpt(path=path, content="# existing code")


class TimeoutProvider(FakeLLMProvider):
    """A provider that always times out, for retry-path tests."""

    def __init__(self) -> None:
        super().__init__(fail_with=LLMTimeoutError(30.0, model="fake-coder"))


def kind_of(result: ToolResult) -> ToolKind:
    return result.kind
