"""A deterministic in-process provider for CI and GPU-less development.

Spec section 45 asks for a fake worker that produces believable planner, coder
and reviewer answers. "Believable" here means *schema-valid*: the default
answers validate against the models in ``structured.py``, the plan graph is
acyclic, and the patch is a real unified diff — otherwise end-to-end tests would
exercise the repair loop instead of the workflow.

Two modes, both deterministic:

*   **Scripted** — a queue of ``ScriptedResponse`` consumed one per call. This is
    how a test asks for slowness, a timeout, a server error, an unparsable
    answer or a truncated one.
*   **Derived** — with an empty queue, the answer is computed from the objective
    found in the prompt. The same prompt always yields byte-identical output, so
    a test can assert on it.

The fake deliberately does *not* emulate a model's unreliability by default.
Tests that need unreliability script it explicitly, so the failure under test is
visible in the test rather than hidden in a random seed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import deque
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from domain.enums import AgentRole, ReviewVerdict
from domain.exceptions import InferenceError, LLMTimeoutError
from domain.ports.llm_provider import LLMProvider, LLMProviderFactory
from domain.value_objects.llm import (
    ChatRole,
    CompletionRequest,
    CompletionResult,
    FinishReason,
    ModelInfo,
    TokenUsage,
    ToolCall,
)
from domain.value_objects.worker import WorkerEndpoint
from infrastructure.llm.structured import OUTPUT_MODELS

__all__ = [
    "FakeLLMProvider",
    "FakeLLMProviderFactory",
    "ScriptedResponse",
]

DEFAULT_FAKE_MODEL: Final = "fake-model"
_DEFAULT_CONTEXT_LENGTH: Final = 32_768
_ROLE_MARKER_RE = re.compile(r"^ROLE:\s*(PLANNER|CODER|REVIEWER)\s*$", re.MULTILINE)
_OBJECTIVE_HEADING_RE = re.compile(r"^#{1,6}\s*objective\b.*$", re.IGNORECASE)
_OBJECTIVE_INLINE_RE = re.compile(r"^\s*objective\s*:\s*(\S.*)$", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ROLE_BY_SCHEMA_TITLE: Final[Mapping[str, AgentRole]] = {
    model.__name__: role for role, model in OUTPUT_MODELS.items()
}
_MAX_OBJECTIVE_CHARS: Final = 200


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    """One programmed answer, or one programmed failure.

    ``content`` of ``None`` means "answer as you normally would", which lets a
    test script only latency or only token usage without restating a whole
    plan.
    """

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage | None = None
    delay_seconds: float = 0.0
    error: Exception | None = None
    chunks: tuple[str, ...] | None = None

    @classmethod
    def text(cls, content: str, **kwargs: Any) -> ScriptedResponse:
        return cls(content=content, **kwargs)

    @classmethod
    def slow(cls, seconds: float, *, content: str | None = None) -> ScriptedResponse:
        """A normal answer that takes time — for deadline and cancellation tests."""
        return cls(content=content, delay_seconds=seconds)

    @classmethod
    def timeout(
        cls, *, after_seconds: float = 0.0, timeout_seconds: float = 1.0
    ) -> ScriptedResponse:
        return cls(
            delay_seconds=after_seconds,
            error=LLMTimeoutError(timeout_seconds, model=DEFAULT_FAKE_MODEL),
        )

    @classmethod
    def server_error(cls, *, status_code: int = 500) -> ScriptedResponse:
        return cls(error=InferenceError("fake inference server failure", status_code=status_code))

    @classmethod
    def invalid_structured_output(
        cls, content: str = "I looked at the repository and the plan is basically fine."
    ) -> ScriptedResponse:
        """Prose where JSON was required: drives the repair loop."""
        return cls(content=content)

    @classmethod
    def truncated(cls, content: str = '{"objective": "build the th') -> ScriptedResponse:
        return cls(content=content, finish_reason=FinishReason.LENGTH)


class FakeLLMProvider:
    """``LLMProvider`` that never leaves the process."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_FAKE_MODEL,
        context_length: int = _DEFAULT_CONTEXT_LENGTH,
        script: Iterable[ScriptedResponse] = (),
        default_role: AgentRole | None = None,
        default_verdict: ReviewVerdict = ReviewVerdict.PASS,
        latency_ms: int = 1,
        chunk_size: int = 64,
        healthy: bool = True,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self._model_info = ModelInfo(model_id=model_id, context_length=context_length)
        self._queue: deque[ScriptedResponse] = deque(script)
        self._default_role = default_role
        self._default_verdict = default_verdict
        self._latency_ms = latency_ms
        self._chunk_size = chunk_size
        self.healthy = healthy
        self._calls: list[CompletionRequest] = []

    # -- inspection used by tests ---------------------------------------
    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def calls(self) -> tuple[CompletionRequest, ...]:
        """Every request received, in order."""
        return tuple(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def script(self, *responses: ScriptedResponse) -> FakeLLMProvider:
        """Queue answers for the next calls. Returns ``self`` so it can chain."""
        self._queue.extend(responses)
        return self

    def reset(self) -> None:
        self._queue.clear()
        self._calls.clear()

    # -- provider -------------------------------------------------------
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self._calls.append(request)
        scripted = self._next()
        await self._wait(scripted)
        content = self._content_of(scripted, request)
        return CompletionResult(
            content=content,
            model=request.model or self._model_info.model_id,
            finish_reason=scripted.finish_reason,
            tool_calls=scripted.tool_calls,
            usage=scripted.usage or _derived_usage(request, content),
            latency_ms=self._latency_ms,
            correlation_id=request.correlation_id,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Same contract as the HTTP provider: an async generator.

        The scripted answer is taken when iteration starts, not when the
        generator object is created, so a script stays aligned with the calls
        that actually happened.
        """
        self._calls.append(request)
        scripted = self._next()
        await self._wait(scripted)
        for chunk in self._chunks_of(scripted, request):
            yield chunk

    async def health(self) -> bool:
        return self.healthy

    # -- internals ------------------------------------------------------
    def _next(self) -> ScriptedResponse:
        """The next programmed answer, or a blank one meaning "derive it"."""
        return self._queue.popleft() if self._queue else ScriptedResponse()

    async def _wait(self, scripted: ScriptedResponse) -> None:
        """Sleep, then fail if programmed to.

        Sleeping first is what makes ``ScriptedResponse.timeout`` usable as a
        cancellation target: the caller's deadline can fire during the sleep.
        """
        if scripted.delay_seconds:
            await asyncio.sleep(scripted.delay_seconds)
        if scripted.error is not None:
            raise scripted.error

    def _content_of(self, scripted: ScriptedResponse, request: CompletionRequest) -> str:
        if scripted.content is not None:
            return scripted.content
        if scripted.tool_calls:
            return ""
        return self.answer_for(request)

    def _chunks_of(self, scripted: ScriptedResponse, request: CompletionRequest) -> Sequence[str]:
        if scripted.chunks is not None:
            return scripted.chunks
        content = self._content_of(scripted, request)
        return [
            content[index : index + self._chunk_size]
            for index in range(0, len(content), self._chunk_size)
        ]

    # -- derived answers ------------------------------------------------
    def answer_for(self, request: CompletionRequest) -> str:
        """The default answer for a request. Pure function of the request."""
        role = self._role_of(request)
        objective = _objective_of(request)
        if role is AgentRole.PLANNER:
            return _planner_answer(objective)
        if role is AgentRole.CODER:
            return _coder_answer(objective)
        return _reviewer_answer(objective, self._default_verdict)

    def _role_of(self, request: CompletionRequest) -> AgentRole:
        """Identify the role from the schema first, the prompt marker second.

        The schema is the stronger signal: it is what the caller actually asked
        to be produced. The ``ROLE:`` marker at the top of every template covers
        unstructured calls.
        """
        if request.json_schema is not None:
            title = str(request.json_schema.get("title", ""))
            by_schema = _ROLE_BY_SCHEMA_TITLE.get(title)
            if by_schema is not None:
                return by_schema
        for message in request.messages:
            marker = _ROLE_MARKER_RE.search(message.content)
            if marker is not None:
                return AgentRole(marker.group(1))
        return self._default_role or AgentRole.PLANNER


class FakeLLMProviderFactory:
    """``LLMProviderFactory`` handing out fakes.

    By default every endpoint shares one provider, so a test scripts the whole
    run in one place and sees the calls in the order the orchestrator made them.
    Pass ``shared=False`` when a test needs per-worker isolation.
    """

    def __init__(
        self,
        provider: FakeLLMProvider | None = None,
        *,
        shared: bool = True,
        model_id: str = DEFAULT_FAKE_MODEL,
        context_length: int = _DEFAULT_CONTEXT_LENGTH,
    ) -> None:
        self._shared = shared
        self._model_id = model_id
        self._context_length = context_length
        self._providers: dict[tuple[str, str], FakeLLMProvider] = {}
        self._default = provider or FakeLLMProvider(
            model_id=model_id, context_length=context_length
        )
        self.requested: list[tuple[str, str]] = []

    @property
    def provider(self) -> FakeLLMProvider:
        """The shared provider: where a test queues its script."""
        return self._default

    @property
    def providers(self) -> Mapping[tuple[str, str], FakeLLMProvider]:
        """Per-(endpoint, model) providers created in isolated mode."""
        return dict(self._providers)

    def script(self, *responses: ScriptedResponse) -> FakeLLMProviderFactory:
        self._default.script(*responses)
        return self

    def for_endpoint(self, endpoint: WorkerEndpoint, *, model_id: str) -> LLMProvider:
        return self.fake_for_endpoint(endpoint, model_id=model_id)

    def fake_for_endpoint(self, endpoint: WorkerEndpoint, *, model_id: str) -> FakeLLMProvider:
        """Typed accessor: tests need the fake, not the port."""
        key = (endpoint.url.rstrip("/"), model_id)
        self.requested.append(key)
        if self._shared:
            return self._default
        provider = self._providers.get(key)
        if provider is None:
            provider = FakeLLMProvider(model_id=model_id, context_length=self._context_length)
            self._providers[key] = provider
        return provider


# -- deterministic answer construction ------------------------------------
def _objective_of(request: CompletionRequest) -> str:
    """Recover the objective from the rendered prompt.

    The templates put it under an ``## Objective`` heading, so that is tried
    first; an inline ``Objective: ...`` line and, failing everything, the first
    line of the last user message keep the fake usable with ad-hoc prompts.
    """
    for message in reversed(request.messages):
        found = _objective_in(message.content)
        if found:
            return found[:_MAX_OBJECTIVE_CHARS]
    for message in reversed(request.messages):
        if message.role is ChatRole.USER:
            for line in message.content.splitlines():
                if line.strip():
                    return line.strip()[:_MAX_OBJECTIVE_CHARS]
    return "unspecified objective"


def _objective_in(text: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        inline = _OBJECTIVE_INLINE_RE.match(line)
        if inline is not None:
            return inline.group(1).strip()
        if _OBJECTIVE_HEADING_RE.match(line):
            for candidate in lines[index + 1 :]:
                if candidate.strip():
                    return candidate.strip()
    return None


def _slug(objective: str) -> str:
    slug = _SLUG_RE.sub("_", objective.lower()).strip("_")[:40].strip("_")
    return slug or "change"


def _digest(objective: str) -> str:
    """Stable short digest: varies with the objective, never with the run."""
    return hashlib.sha256(objective.encode("utf-8")).hexdigest()[:8]


def _dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _planner_answer(objective: str) -> str:
    slug = _slug(objective)
    module = f"src/{slug}.py"
    return _dumps(
        {
            "objective": objective,
            "assumptions": [
                "the repository builds and its test suite passes before this change",
            ],
            "constraints": ["do not modify unrelated modules"],
            "tasks": [
                {
                    "key": "analyze",
                    "title": f"Locate the code affected by: {objective}",
                    "description": "Identify the modules to change and the tests covering them.",
                    "target_paths": [module],
                    "depends_on": [],
                    "validation": [],
                },
                {
                    "key": "implement",
                    "title": f"Implement: {objective}",
                    "description": "Apply the change identified by the analysis task.",
                    "target_paths": [module],
                    "depends_on": ["analyze"],
                    "validation": ["pytest -q"],
                },
                {
                    "key": "cover",
                    "title": "Cover the change with tests",
                    "description": "Add focused tests for the new behaviour.",
                    "target_paths": [f"tests/test_{slug}.py"],
                    "depends_on": ["analyze"],
                    "validation": ["pytest -q"],
                },
            ],
            "validation_requirements": ["pytest -q"],
            "risk_areas": ["callers of the modified module"],
        }
    )


def _coder_answer(objective: str) -> str:
    slug = _slug(objective)
    path = f"src/{slug}.py"
    diff = "\n".join(
        (
            f"diff --git a/{path} b/{path}",
            "new file mode 100644",
            "index 0000000..1111111",
            "--- /dev/null",
            f"+++ b/{path}",
            "@@ -0,0 +1,5 @@",
            f'+"""{objective}."""',
            "+",
            "+",
            f"+def {slug[:30] or 'run'}() -> str:",
            f'+    return "{_digest(objective)}"',
            "",
        )
    )
    return _dumps(
        {
            "task_key": "implement",
            "summary": f"Added {path} implementing: {objective}",
            "diff": diff,
            "files_changed": [path],
            "commands_run": ["pytest -q"],
            "uncertainties": [],
        }
    )


def _reviewer_answer(objective: str, verdict: ReviewVerdict) -> str:
    if verdict is ReviewVerdict.PASS:
        return _dumps(
            {
                "verdict": "PASS",
                "summary": f"The patch implements the task and validation is green: {objective}",
                "findings": [],
            }
        )
    return _dumps(
        {
            "verdict": "FAIL",
            "summary": f"The patch does not fully implement: {objective}",
            "findings": [
                {
                    "summary": "The objective is not covered by a test",
                    "severity": "MAJOR",
                    "file": f"src/{_slug(objective)}.py",
                    "line": 1,
                    "repair_instruction": "Add a focused test exercising the new behaviour.",
                }
            ],
        }
    )


def _derived_usage(request: CompletionRequest, content: str) -> TokenUsage:
    """Four characters per token: wrong, but stable, which is what tests need."""
    prompt_chars = sum(len(message.content) for message in request.messages)
    return TokenUsage(input_tokens=prompt_chars // 4, output_tokens=len(content) // 4)


def _port_conformance(
    provider: FakeLLMProvider, factory: FakeLLMProviderFactory
) -> tuple[LLMProvider, LLMProviderFactory]:
    """Static-only guard: the fake must satisfy the same ports as the real one."""
    return provider, factory
