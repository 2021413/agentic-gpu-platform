"""Application-level ports.

Prompt rendering and structured-output decoding are application concerns whose
*implementations* are infrastructure (a template loader, a pydantic schema
library). Declaring them here keeps the orchestration testable with fakes and
preserves the dependency rule: application never imports infrastructure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from application.dto.agent_io import CodeDraft, PlanDraft, ReviewDraft
from domain.entities.project import Project
from domain.enums import AgentRole
from domain.ports.repositories import UnitOfWork
from domain.ports.tools import ToolExecutor
from domain.value_objects.identifiers import RunId
from domain.value_objects.llm import ChatMessage

__all__ = [
    "AgentOutputCodec",
    "MetricsRecorder",
    "PromptRenderer",
    "RenderedPrompt",
    "RunCoordinator",
    "ToolExecutorFactory",
    "UnitOfWorkFactory",
]


UnitOfWorkFactory = Callable[[], UnitOfWork]
"""Builds a fresh unit of work per transaction.

A single shared instance would serialize — or worse, interleave — concurrent
requests on one database session. Every use case opens its own.
"""


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt ready to send, plus the template version that produced it.

    The version is stored on the job: without it, a past run cannot be
    explained, let alone reproduced (spec section 29).
    """

    messages: tuple[ChatMessage, ...]
    version: str
    role: AgentRole

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a rendered prompt must contain at least one message")


@runtime_checkable
class PromptRenderer(Protocol):
    """Loads versioned prompt templates and renders them with variables."""

    def render(
        self,
        *,
        role: AgentRole,
        variables: Mapping[str, Any],
        version: str | None = None,
    ) -> RenderedPrompt:
        """Render the template for ``role``; ``None`` means the current version."""
        ...

    def current_version(self, role: AgentRole) -> str: ...


@runtime_checkable
class AgentOutputCodec(Protocol):
    """Validates model answers into the drafts the orchestrator can act on.

    Implementations must raise ``StructuredOutputError`` rather than returning a
    half-parsed value: a malformed answer is a failure with its own retry rule,
    not a value to guess at.
    """

    def schema_for(self, role: AgentRole) -> Mapping[str, Any]:
        """JSON schema to attach to the completion request."""
        ...

    def parse_plan(self, raw: str) -> PlanDraft: ...
    def parse_code(self, raw: str) -> CodeDraft: ...
    def parse_review(self, raw: str) -> ReviewDraft: ...

    def repair_messages(self, *, role: AgentRole, raw: str, error: str) -> Sequence[ChatMessage]:
        """Follow-up turns re-asking the model with the violation attached."""
        ...


@runtime_checkable
class MetricsRecorder(Protocol):
    """Counters and histograms (spec section 32).

    Telemetry must never be mandatory for unit tests, so every method is
    fire-and-forget and a no-op implementation is always acceptable.
    """

    def increment(self, name: str, value: int = 1, **labels: str) -> None: ...
    def observe(self, name: str, value: float, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...


@runtime_checkable
class RunCoordinator(Protocol):
    """Serializes the decisions taken about one run (spec section 27).

    Candidates run concurrently on purpose, but the moment they report back they
    all want to answer the same question — "are we done, and who won?". That
    decision must be taken once. Only the run's own decisions are serialized;
    the candidates themselves never contend.

    A single orchestrator process is served by an in-process implementation; as
    soon as several orchestrators share a project, this must be backed by a
    distributed lock.
    """

    def lock(self, run_id: RunId) -> AbstractAsyncContextManager[None]:
        """Held for the duration of one run decision."""
        ...


@runtime_checkable
class ToolExecutorFactory(Protocol):
    """Builds the tool set a role may use on a given project.

    Which tools exist is a property of the project, not of the platform: the
    build and test commands come from its toolchain, and a project that
    configures none simply has no such tool. Handing the orchestrator one
    global executor would have forced it to invent commands instead.
    """

    def for_project(self, project: Project, *, role: AgentRole) -> ToolExecutor:
        """Executor restricted to what ``role`` may run on ``project``."""
        ...

    def available_tools(self, project: Project, *, role: AgentRole) -> Sequence[str]:
        """Tool names available, so a stage with no command can be skipped."""
        ...
