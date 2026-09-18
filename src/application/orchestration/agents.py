"""The three agent roles, expressed over ports (spec sections 6 and 30).

An agent turns context into a *validated* draft. It never touches the
filesystem, never decides workflow state, and never returns unparsed prose: a
model answer that does not satisfy its schema is re-asked once with the
violation attached, then reported as a failure with its own retry semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from application.dto.agent_io import CodeDraft, PlanDraft, ReviewDraft, format_findings
from application.ports import AgentOutputCodec, PromptRenderer
from domain.entities.candidate import Candidate
from domain.entities.plan import Plan
from domain.entities.project import Project
from domain.entities.review import Review
from domain.entities.run import Run
from domain.enums import AgentRole
from domain.exceptions import StructuredOutputError
from domain.ports.llm_provider import LLMProvider
from domain.ports.repository_context import RepositoryContext
from domain.value_objects.llm import ChatMessage, CompletionRequest, TokenUsage
from domain.value_objects.validation import ValidationReport

__all__ = [
    "AgentOutcome",
    "CoderAgent",
    "PlannerAgent",
    "ReviewerAgent",
    "StructuredCompletion",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AgentOutcome[TDraft]:
    """A validated draft plus everything needed to explain how it was produced."""

    draft: TDraft
    usage: TokenUsage
    prompt_version: str
    model: str
    repairs: int = 0
    raw_length: int = 0


@dataclass(frozen=True, slots=True)
class StructuredCompletion:
    """Shared machinery: render a versioned prompt, complete, validate, repair.

    Composed by all three agents rather than inherited, so a role adds context
    without inheriting a behaviour it did not ask for.
    """

    renderer: PromptRenderer
    codec: AgentOutputCodec
    max_structured_output_repairs: int = 2
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    extra_variables: Mapping[str, Any] = field(default_factory=dict)

    async def complete(
        self,
        *,
        provider: LLMProvider,
        role: AgentRole,
        variables: Mapping[str, Any],
        parse: Callable[[str], T],
    ) -> AgentOutcome[T]:
        """Complete, validate, and re-ask once per failure within the budget.

        The repair turns are appended to the same conversation so the model sees
        its own invalid answer and the exact violation, which is far more
        effective than re-sending the original prompt unchanged.
        """
        prompt = self.renderer.render(
            role=role, variables={**dict(self.extra_variables), **dict(variables)}
        )
        messages: list[ChatMessage] = list(prompt.messages)
        schema = self.codec.schema_for(role)
        usage = TokenUsage()
        model = ""
        last_error = ""
        raw = ""

        for attempt in range(self.max_structured_output_repairs + 1):
            result = await provider.complete(
                CompletionRequest(
                    messages=tuple(messages),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_schema=schema,
                    timeout_seconds=self.timeout_seconds,
                )
            )
            usage = usage + result.usage
            model = result.model
            raw = result.content

            if result.truncated:
                # A length-truncated answer can never be valid JSON; re-asking
                # the same way would only burn tokens.
                raise StructuredOutputError(
                    "model output was truncated by the token limit",
                    schema=str(role),
                    raw_output=raw,
                    attempt=attempt + 1,
                )
            try:
                return AgentOutcome(
                    draft=parse(raw),
                    usage=usage,
                    prompt_version=prompt.version,
                    model=model,
                    repairs=attempt,
                    raw_length=len(raw),
                )
            except (StructuredOutputError, ValueError) as exc:
                last_error = str(exc)
                if attempt >= self.max_structured_output_repairs:
                    break
                messages.extend(self.codec.repair_messages(role=role, raw=raw, error=last_error))

        raise StructuredOutputError(
            f"structured output still invalid after {self.max_structured_output_repairs} "
            f"repair attempt(s): {last_error}",
            schema=str(role),
            raw_output=raw,
            attempt=self.max_structured_output_repairs + 1,
        )


@dataclass(frozen=True, slots=True)
class PlannerAgent:
    """Turns an objective into a validated, acyclic task graph.

    The planner is read-only by construction: it is never handed a writable
    workspace, so it cannot mutate project files even if it asks to.
    """

    completion: StructuredCompletion

    async def plan(
        self,
        *,
        provider: LLMProvider,
        project: Project,
        run: Run,
        context: RepositoryContext,
        previous_plan: Plan | None = None,
        revision_reason: str | None = None,
    ) -> AgentOutcome[PlanDraft]:
        return await self.completion.complete(
            provider=provider,
            role=AgentRole.PLANNER,
            variables={
                "objective": run.objective,
                "project_name": project.name,
                "language": project.toolchain.language,
                "build_command": project.toolchain.build_command or "none configured",
                "test_command": project.toolchain.test_command or "none configured",
                "repository_context": context.render(),
                "revision": run.plan_revisions,
                "previous_plan": _render_previous_plan(previous_plan),
                "revision_reason": revision_reason or "",
            },
            parse=self.completion.codec.parse_plan,
        )


@dataclass(frozen=True, slots=True)
class CoderAgent:
    """Produces a candidate patch for one workspace."""

    completion: StructuredCompletion

    async def code(
        self,
        *,
        provider: LLMProvider,
        project: Project,
        run: Run,
        candidate: Candidate,
        context: RepositoryContext,
        plan: Plan | None = None,
        repair_brief: str | None = None,
        tool_output: str = "",
    ) -> AgentOutcome[CodeDraft]:
        return await self.completion.complete(
            provider=provider,
            role=AgentRole.CODER,
            variables={
                "objective": run.objective,
                "project_name": project.name,
                "language": project.toolchain.language,
                "build_command": project.toolchain.build_command or "none configured",
                "test_command": project.toolchain.test_command or "none configured",
                "repository_context": context.render(),
                "plan": _render_plan(plan),
                "candidate_index": candidate.index,
                "iteration": candidate.coder_iterations,
                "repair_brief": repair_brief or "",
                "tool_output": tool_output,
            },
            parse=self.completion.codec.parse_code,
        )


@dataclass(frozen=True, slots=True)
class ReviewerAgent:
    """Judges one candidate on its patch and its deterministic evidence.

    It deliberately does not receive the coder's conversation: reviewing the
    reasoning that produced a defect is how a reviewer inherits it.
    """

    completion: StructuredCompletion

    async def review(
        self,
        *,
        provider: LLMProvider,
        project: Project,
        run: Run,
        candidate: Candidate,
        validation: ValidationReport,
        plan: Plan | None = None,
        previous_reviews: Sequence[Review] = (),
    ) -> AgentOutcome[ReviewDraft]:
        patch = candidate.patch
        return await self.completion.complete(
            provider=provider,
            role=AgentRole.REVIEWER,
            variables={
                "objective": run.objective,
                "project_name": project.name,
                "language": project.toolchain.language,
                "plan": _render_plan(plan),
                "diff": patch.diff if patch else "",
                "changed_files": ", ".join(patch.changed_paths) if patch else "",
                "implementation_summary": candidate.summary,
                "uncertainties": "\n".join(f"- {u}" for u in candidate.uncertainties),
                "validation_summary": validation.summary(),
                "validation_details": "\n\n".join(
                    f"$ {r.command}\nexit={r.exit_code}\n{r.tail(2000)}" for r in validation.results
                ),
                "previous_findings": "\n".join(r.repair_brief() for r in previous_reviews),
            },
            parse=self.completion.codec.parse_review,
        )


def _render_plan(plan: Plan | None) -> str:
    if plan is None:
        return "No plan: this objective was judged simple enough to implement directly."
    lines = [f"Objective: {plan.objective}", "Tasks:"]
    for task in plan.tasks:
        deps = f" (after {', '.join(task.depends_on)})" if task.depends_on else ""
        lines.append(f"- {task.key}: {task.title}{deps}")
        if task.description:
            lines.append(f"  {task.description}")
    if plan.constraints:
        lines.append("Constraints:")
        lines.extend(f"- {c}" for c in plan.constraints)
    return "\n".join(lines)


def _render_previous_plan(plan: Plan | None) -> str:
    if plan is None:
        return ""
    return "The previous revision was rejected:\n" + _render_plan(plan)


def render_repair_brief(review: ReviewDraft) -> str:
    """Reviewer findings as the instructions a coder receives on repair."""
    return format_findings(review.findings)
