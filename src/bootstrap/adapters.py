"""Adapters between application ports and concrete infrastructure.

The application says what it needs semantically ("render the planner prompt for
this objective, in this project, with this context"); a prompt template speaks
its own vocabulary. Translating between the two is exactly what an adapter is
for, and putting it here means neither side has to know the other's names.

This module lives in the composition root because it is the only place allowed
to see both an application port and an infrastructure class at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from application.dto.agent_io import (
    CodeDraft,
    FindingDraft,
    PlanDraft,
    ReviewDraft,
    TaskDraft,
)
from application.ports import RenderedPrompt
from domain.enums import AgentRole
from domain.ports.tools import ToolRegistry
from domain.value_objects.llm import ChatMessage
from infrastructure.llm.prompts import PromptLibrary
from infrastructure.llm.structured import (
    CoderOutput,
    PlannerOutput,
    ReviewerOutput,
    StructuredOutputParser,
)

__all__ = ["PromptLibraryRenderer", "StructuredOutputCodec"]

_ANSWER_NOW = "Produce your answer now, as a single JSON object matching the schema."


class StructuredOutputCodec:
    """Validates model answers with pydantic, then hands back plain drafts.

    The application must not depend on pydantic models defined in
    infrastructure, so the validated object is converted here. The schemas
    replay the domain's invariants — an acyclic task graph, a FAIL carrying
    findings — which turns a violation into a repair prompt instead of an
    exception three layers later.
    """

    __slots__ = ("_parsers",)

    def __init__(self, *, max_repair_attempts: int = 2) -> None:
        self._parsers: dict[AgentRole, StructuredOutputParser[Any]] = {
            AgentRole.PLANNER: StructuredOutputParser(
                PlannerOutput, max_repair_attempts=max_repair_attempts
            ),
            AgentRole.CODER: StructuredOutputParser(
                CoderOutput, max_repair_attempts=max_repair_attempts
            ),
            AgentRole.REVIEWER: StructuredOutputParser(
                ReviewerOutput, max_repair_attempts=max_repair_attempts
            ),
        }

    def schema_for(self, role: AgentRole) -> Mapping[str, Any]:
        return self._parsers[role].json_schema

    def schema_text(self, role: AgentRole) -> str:
        return self._parsers[role].schema_text()

    def parse_plan(self, raw: str) -> PlanDraft:
        output: PlannerOutput = self._parsers[AgentRole.PLANNER].parse(raw)
        return PlanDraft(
            objective=output.objective,
            tasks=tuple(
                TaskDraft(
                    key=task.key,
                    title=task.title,
                    description=task.description,
                    depends_on=tuple(task.depends_on),
                    target_paths=tuple(task.target_paths),
                    validation=tuple(task.validation),
                )
                for task in output.tasks
            ),
            assumptions=tuple(output.assumptions),
            constraints=tuple(output.constraints),
            risk_areas=tuple(output.risk_areas),
            validation_requirements=tuple(output.validation_requirements),
        )

    def parse_code(self, raw: str) -> CodeDraft:
        output: CoderOutput = self._parsers[AgentRole.CODER].parse(raw)
        return CodeDraft(
            summary=output.summary,
            diff=output.diff,
            uncertainties=tuple(output.uncertainties),
        )

    def parse_review(self, raw: str) -> ReviewDraft:
        output: ReviewerOutput = self._parsers[AgentRole.REVIEWER].parse(raw)
        return ReviewDraft(
            verdict=output.verdict,
            summary=output.summary,
            findings=tuple(
                FindingDraft(
                    summary=f.summary,
                    severity=f.severity,
                    file=f.file,
                    line=f.line,
                    repair_instruction=f.repair_instruction,
                )
                for f in output.findings
            ),
        )

    def repair_messages(self, *, role: AgentRole, raw: str, error: str) -> Sequence[ChatMessage]:
        """Re-ask with the rejected answer and the exact violation attached.

        Showing the model its own output is what makes the second attempt work;
        re-sending the original prompt unchanged usually reproduces the error.
        """
        return (
            ChatMessage.assistant(raw),
            ChatMessage.user(
                f"That answer was rejected: {error}\n\n"
                f"Required schema:\n{self.schema_text(role)}\n\n{_ANSWER_NOW}"
            ),
        )


class PromptLibraryRenderer:
    """Renders the application's variables into the versioned templates.

    The mapping below is the whole point of the class: the orchestrator thinks
    in terms of objective, plan and validation evidence, while a template asks
    for ``project_context``, ``task`` and ``validation_report``. Keeping the
    translation explicit means a template can be rewritten without touching a
    single use case.
    """

    __slots__ = ("_codec", "_library", "_tools")

    def __init__(
        self,
        *,
        library: PromptLibrary,
        codec: StructuredOutputCodec,
        tools: Mapping[AgentRole, ToolRegistry] | None = None,
    ) -> None:
        self._library = library
        self._codec = codec
        self._tools = dict(tools or {})

    def current_version(self, role: AgentRole) -> str:
        return self._library.default_version(role)

    def render(
        self,
        *,
        role: AgentRole,
        variables: Mapping[str, Any],
        version: str | None = None,
    ) -> RenderedPrompt:
        rendered = self._library.render(
            role, self._template_variables(role, variables), version=version
        )
        return RenderedPrompt(
            messages=(
                ChatMessage.system(rendered.text),
                ChatMessage.user(_ANSWER_NOW),
            ),
            version=rendered.version,
            role=role,
        )

    # ------------------------------------------------------------------
    def _template_variables(
        self, role: AgentRole, variables: Mapping[str, Any]
    ) -> dict[str, object]:
        get = variables.get
        common = {
            "objective": str(get("objective", "")),
            "allowed_tools": self._allowed_tools(role),
            "output_schema": self._codec.schema_text(role),
        }
        if role is AgentRole.PLANNER:
            return {
                **common,
                "project_context": _project_context(variables),
                "constraints": _bullets(get("constraints")) or "None stated.",
                "retry_context": _join_sections(
                    ("Reason for this revision", get("revision_reason")),
                    ("Previous plan", get("previous_plan")),
                ),
            }
        if role is AgentRole.CODER:
            return {
                **common,
                "task": str(get("plan") or "Implement the objective directly."),
                "workspace_context": _project_context(variables),
                "retry_context": _join_sections(
                    ("Reviewer findings to address", get("repair_brief")),
                    ("Recent tool output", get("tool_output")),
                ),
            }
        return {
            **common,
            "task": str(get("plan") or "Implement the objective directly."),
            "patch": _patch_section(variables),
            "validation_report": _join_sections(
                ("Summary", get("validation_summary")),
                ("Details", get("validation_details")),
                ("Implementation summary", get("implementation_summary")),
                ("Stated uncertainties", get("uncertainties")),
            )
            or "No deterministic validation was run.",
            "retry_context": _join_sections(
                ("Findings from earlier reviews", get("previous_findings"))
            ),
        }

    def _allowed_tools(self, role: AgentRole) -> str:
        registry = self._tools.get(role)
        if registry is None:
            return "none"
        names = registry.names()
        return ", ".join(names) if names else "none"


def _project_context(variables: Mapping[str, Any]) -> str:
    get = variables.get
    header = "\n".join(
        (
            f"Project: {get('project_name', 'unknown')}",
            f"Language: {get('language', 'unknown')}",
            f"Build command: {get('build_command', 'none configured')}",
            f"Test command: {get('test_command', 'none configured')}",
        )
    )
    context = str(get("repository_context", "")).strip()
    return f"{header}\n\n{context}" if context else header


def _patch_section(variables: Mapping[str, Any]) -> str:
    diff = str(variables.get("diff", "")).strip()
    files = str(variables.get("changed_files", "")).strip()
    if not diff:
        return "The candidate produced no changes."
    return f"Changed files: {files}\n\n{diff}" if files else diff


def _bullets(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def _join_sections(*sections: tuple[str, object]) -> str:
    """Render only the sections that carry something.

    Empty headings are worse than absent ones: they invite the model to invent
    content for a section the orchestrator had nothing to put in.
    """
    rendered = [
        f"## {title}\n{_bullets(body).strip()}"
        for title, body in sections
        if _bullets(body).strip()
    ]
    return "\n\n".join(rendered)
