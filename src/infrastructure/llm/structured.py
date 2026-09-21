"""Structured outputs for every control decision (spec section 30).

The rule this module exists to enforce: no orchestration decision is ever read
out of prose. A model answer is either a document that validates against a
Pydantic schema, or it is not an answer at all — in which case the model is told
precisely what was wrong, a bounded number of times, and then the job fails
explicitly.

The Pydantic models mirror the domain aggregates they feed (``Plan``,
``Patch``, ``Review``) and repeat their invariants — an acyclic task graph, a
failing review that carries findings. Repeating them here is deliberate: a
violation caught at this boundary becomes a repair prompt the model can act on,
whereas the same violation caught in the domain is an unrecoverable error.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from domain.entities.review import ReviewFinding, Severity
from domain.enums import AgentRole, ReviewVerdict
from domain.exceptions import StructuredOutputError
from domain.ports.llm_provider import LLMProvider
from domain.value_objects.llm import ChatMessage, CompletionRequest, CompletionResult, TokenUsage

__all__ = [
    "OUTPUT_MODELS",
    "CoderOutput",
    "PlannedTask",
    "PlannerOutput",
    "ReviewerFinding",
    "ReviewerOutput",
    "StructuredCompletion",
    "StructuredOutputParser",
    "extract_json_object",
]

_MAX_RAW_EXCERPT: Final = 2_000
_REASONING_PAIR_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_REASONING_CLOSE_RE = re.compile(r"</(?:think|thinking|reasoning)>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)(?:```|\Z)", re.DOTALL)


class _StrictModel(BaseModel):
    """Shared configuration.

    Unknown keys are ignored rather than rejected: models routinely decorate
    their answer with an extra field, and spending another GPU generation on
    that would be waste. Missing or malformed *required* data still fails.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class PlannedTask(_StrictModel):
    """One unit of work, named by a planner-chosen stable key."""

    key: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1)
    description: str = ""
    target_paths: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def _key_is_a_slug(cls, value: str) -> str:
        """Keys are cross-referenced by other tasks; whitespace makes that fragile."""
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
            raise ValueError("task key must be a slug: letters, digits, '-', '_' or '.', no spaces")
        return value


class PlannerOutput(_StrictModel):
    """Planner answer (spec section 6)."""

    objective: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tasks: list[PlannedTask] = Field(min_length=1)
    validation_requirements: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _graph_is_usable(self) -> PlannerOutput:
        """Reject duplicate keys, dangling dependencies and cycles.

        A cyclic plan would deadlock the scheduler; caught here, it is just a
        repair prompt away from a usable plan.
        """
        keys = [task.key for task in self.tasks]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate task keys: {duplicates}")

        known = set(keys)
        for task in self.tasks:
            if task.key in task.depends_on:
                raise ValueError(f"task {task.key!r} depends on itself")
            unknown = sorted(set(task.depends_on) - known)
            if unknown:
                raise ValueError(f"task {task.key!r} depends on unknown tasks: {unknown}")

        remaining = {task.key: set(task.depends_on) for task in self.tasks}
        while remaining:
            ready = [key for key, deps in remaining.items() if not deps]
            if not ready:
                raise ValueError(f"task dependencies contain a cycle: {sorted(remaining)}")
            for key in ready:
                del remaining[key]
            for deps in remaining.values():
                deps.difference_update(ready)
        return self

    def execution_layers(self) -> tuple[tuple[str, ...], ...]:
        """Task keys grouped into concurrently executable layers."""
        remaining = {task.key: set(task.depends_on) for task in self.tasks}
        layers: list[tuple[str, ...]] = []
        while remaining:
            ready = tuple(key for key, deps in remaining.items() if not deps)
            layers.append(ready)
            for key in ready:
                del remaining[key]
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(layers)


class FileEdit(_StrictModel):
    """One file written in full.

    Preferred over a diff: a model writes a file reliably, while a unified diff
    demands exact hunk headers and line counts it gets wrong often enough to
    fail runs outright. git computes the diff afterwards, from the truth on
    disk.
    """

    path: str = Field(min_length=1)
    content: str

    @field_validator("path")
    @classmethod
    def _stays_inside_the_workspace(cls, value: str) -> str:
        """Refuse traversal here as well as at the filesystem boundary.

        Defence in depth: the workspace manager checks containment too, but a
        path escaping the workspace should never even reach it.
        """
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"path must be relative and stay inside the workspace: {value!r}")
        return value


class CoderOutput(_StrictModel):
    """Coder answer: the change, plus what the coder is unsure about."""

    task_key: str | None = None
    summary: str = Field(min_length=1)
    files: list[FileEdit] = Field(default_factory=list)
    diff: str = ""
    files_changed: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _expresses_a_change(self) -> CoderOutput:
        """One of the two forms must carry something."""
        if not self.files and not self.diff.strip():
            raise ValueError(
                "answer with 'files': [{'path': ..., 'content': ...}] containing the "
                "full new content of each file you change"
            )
        return self

    @field_validator("diff")
    @classmethod
    def _looks_like_a_unified_diff(cls, value: str) -> str:
        """Cheap shape check only.

        Whether the patch *applies* is decided by the deterministic patch tool,
        never here. This catches the common failure of a model describing its
        change in prose where a diff was required.
        """
        if not value.strip():
            return value
        if "diff --git " not in value and not re.search(r"^@@ .+ @@", value, re.MULTILINE):
            raise ValueError(
                "diff must be a unified diff containing 'diff --git' or '@@' hunk headers"
            )
        return value


class ReviewerFinding(_StrictModel):
    """One defect, with the instruction that lets a coder repair it."""

    summary: str = Field(min_length=1)
    severity: Severity = Severity.MAJOR
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    repair_instruction: str | None = None


class ReviewerOutput(_StrictModel):
    """Reviewer verdict (spec section 6)."""

    verdict: ReviewVerdict
    summary: str = ""
    findings: list[ReviewerFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _failure_is_actionable(self) -> ReviewerOutput:
        """``Review`` refuses a FAIL without findings; catch it while repairable."""
        if self.verdict is ReviewVerdict.FAIL and not self.findings:
            raise ValueError("a FAIL verdict must list at least one finding")
        return self

    def to_findings(self) -> tuple[ReviewFinding, ...]:
        return tuple(
            ReviewFinding(
                summary=finding.summary,
                severity=finding.severity,
                file=finding.file,
                line=finding.line,
                repair_instruction=finding.repair_instruction,
            )
            for finding in self.findings
        )


OUTPUT_MODELS: Final[Mapping[AgentRole, type[BaseModel]]] = {
    AgentRole.PLANNER: PlannerOutput,
    AgentRole.CODER: CoderOutput,
    AgentRole.REVIEWER: ReviewerOutput,
}


@dataclass(frozen=True, slots=True)
class StructuredCompletion[ModelT: BaseModel]:
    """A validated answer plus what it cost to obtain it.

    ``attempts`` and ``usage`` cover the repair round-trips too: a model that
    needs three attempts to emit valid JSON is a real cost, and hiding it would
    make the token accounting lie.
    """

    value: ModelT
    result: CompletionResult
    raw_output: str
    attempts: int
    usage: TokenUsage


class StructuredOutputParser[ModelT: BaseModel]:
    """Extracts, validates, and — within bounds — repairs a structured answer."""

    def __init__(self, model: type[ModelT], *, max_repair_attempts: int = 2) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative")
        self._model = model
        self._max_repair_attempts = max_repair_attempts

    @property
    def model(self) -> type[ModelT]:
        return self._model

    @property
    def schema_name(self) -> str:
        return self._model.__name__

    @property
    def max_repair_attempts(self) -> int:
        return self._max_repair_attempts

    @property
    def json_schema(self) -> Mapping[str, Any]:
        """The schema handed to the server for constrained decoding."""
        return self._model.model_json_schema()

    def schema_text(self) -> str:
        """The same schema, formatted for embedding in a prompt template."""
        return json.dumps(self.json_schema, indent=2, sort_keys=True)

    def parse(self, text: str) -> ModelT:
        """Validate one answer. Raises ``StructuredOutputError`` on any failure."""
        payload = extract_json_object(text)
        if payload is None:
            raise StructuredOutputError(
                "no JSON object could be extracted from the model answer",
                schema=self.schema_name,
                raw_output=text,
            )
        try:
            return self._model.model_validate(payload)
        except ValidationError as exc:
            raise StructuredOutputError(
                _render_validation_error(exc),
                schema=self.schema_name,
                raw_output=text,
            ) from exc

    def repair_prompt(self, raw_output: str, error: str) -> str:
        """The message sent back to the model after an invalid answer.

        It names the exact validation failure — a generic "that was wrong"
        makes the second attempt no likelier to succeed than the first.
        """
        excerpt = raw_output[:_MAX_RAW_EXCERPT]
        truncation_note = " (truncated)" if len(raw_output) > _MAX_RAW_EXCERPT else ""
        return (
            "Your previous answer was rejected: it does not satisfy the required "
            f"{self.schema_name} schema.\n\n"
            f"Validation error:\n{error}\n\n"
            f"What you returned{truncation_note}:\n{excerpt}\n\n"
            "Answer again with a single JSON object that satisfies the schema below. "
            "Output JSON only: no prose, no markdown fences, no reasoning, no apology.\n\n"
            f"Schema:\n{self.schema_text()}"
        )

    async def complete(
        self, provider: LLMProvider, request: CompletionRequest
    ) -> StructuredCompletion[ModelT]:
        """Run the request until it validates, repairing at most N times.

        Inference errors (timeout, server failure) are *not* repaired here: they
        are not the model's mistake and belong to the retry policy, which may
        move the job to another worker.
        """
        base = (
            request
            if request.json_schema is not None
            else replace(request, json_schema=self.json_schema)
        )
        messages = list(base.messages)
        usage = TokenUsage()

        for attempt in range(1, self._max_repair_attempts + 2):
            result = await provider.complete(base.with_messages(messages))
            usage = usage + result.usage
            raw = _answer_text(result)

            if result.truncated:
                # Retrying with the same budget would truncate identically.
                raise StructuredOutputError(
                    "model answer hit the token limit before the JSON was complete",
                    schema=self.schema_name,
                    raw_output=raw,
                    attempt=attempt,
                )
            try:
                value = self.parse(raw)
            except StructuredOutputError as exc:
                if attempt > self._max_repair_attempts:
                    raise StructuredOutputError(
                        f"model answer still invalid after {attempt} attempts: {exc.message}",
                        schema=self.schema_name,
                        raw_output=raw,
                        attempt=attempt,
                    ) from exc
                messages = [
                    *messages,
                    ChatMessage.assistant(raw),
                    ChatMessage.user(self.repair_prompt(raw, exc.message)),
                ]
                continue
            return StructuredCompletion(
                value=value,
                result=result,
                raw_output=raw,
                attempts=attempt,
                usage=usage,
            )

        raise AssertionError("unreachable: the repair loop always returns or raises")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the JSON object in a model answer, whatever it is wrapped in.

    Tried in order: the whole answer, fenced code blocks, then brace-balanced
    spans. The scan is a parser, not a pattern over prose — it tracks string
    literals and escapes so that a brace inside a diff or a message cannot end
    the object early.
    """
    cleaned = _strip_reasoning(text)
    for candidate in _iter_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _iter_candidates(text: str) -> Iterator[str]:
    stripped = text.strip()
    if stripped:
        yield stripped
    for match in _FENCE_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            yield block
    yield from _balanced_spans(text)


def _balanced_spans(text: str) -> Iterator[str]:
    """Yield every top-level ``{...}`` span, longest-first within the text."""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]


def _strip_reasoning(text: str) -> str:
    """Drop reasoning blocks before looking for JSON (spec section 21).

    Hidden reasoning must never reach a client, and it must never be mistaken
    for the answer either: a ``<think>`` block frequently contains a *draft*
    JSON document that differs from the final one.
    """
    without_pairs = _REASONING_PAIR_RE.sub("", text)
    closings = list(_REASONING_CLOSE_RE.finditer(without_pairs))
    if closings:
        return without_pairs[closings[-1].end() :]
    return without_pairs


def _answer_text(result: CompletionResult) -> str:
    """The text to validate: visible content, or the tool-call arguments.

    Some servers answer a schema-constrained request as a single tool call
    whose arguments *are* the document.
    """
    if result.content.strip():
        return result.content
    if result.tool_calls:
        return result.tool_calls[0].arguments
    return result.content


def _render_validation_error(error: ValidationError) -> str:
    """One line per problem, addressed by JSON path, without Pydantic's URLs."""
    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)
