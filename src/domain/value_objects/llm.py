"""Provider-neutral inference value objects.

Nothing here mentions vLLM, OpenAI or Qwen: those are infrastructure details.
The domain only knows that *something* can turn a list of messages into a
structured answer, and reports how many tokens it cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum, unique
from types import MappingProxyType
from typing import Any

__all__ = [
    "ChatMessage",
    "ChatRole",
    "CompletionRequest",
    "CompletionResult",
    "FinishReason",
    "ModelInfo",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@unique
class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@unique
class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn of a conversation handed to an ``LLMProvider``."""

    role: ChatRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role is ChatRole.TOOL and not self.tool_call_id:
            raise ValueError("a tool message must reference the tool_call_id it answers")

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(ChatRole.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(ChatRole.USER, content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(ChatRole.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A deterministic tool advertised to the model, described by a JSON schema."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model. ``arguments`` stays raw JSON text."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting, aggregated across the jobs of a run."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must not be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What a provider says about the model it serves."""

    model_id: str
    context_length: int
    supports_tools: bool = True
    supports_json_schema: bool = True

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """An inference request, fully described and free of transport concerns.

    ``json_schema`` carries the expectation from spec section 30: control
    decisions are structured outputs, never regex over prose.
    """

    messages: tuple[ChatMessage, ...]
    model: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    json_schema: Mapping[str, Any] | None = None
    tools: tuple[ToolSpec, ...] = ()
    stop: tuple[str, ...] = ()
    seed: int | None = None
    timeout_seconds: float | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a completion request needs at least one message")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be within [0, 2]")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")

    def with_messages(self, messages: Sequence[ChatMessage]) -> CompletionRequest:
        """Return a copy carrying a different conversation (used by repair prompts)."""
        return replace(self, messages=tuple(messages))


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """The provider's answer, plus the accounting the orchestrator persists."""

    content: str
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0
    correlation_id: str | None = None

    @property
    def truncated(self) -> bool:
        """A length-truncated answer can never be valid structured output."""
        return self.finish_reason is FinishReason.LENGTH
