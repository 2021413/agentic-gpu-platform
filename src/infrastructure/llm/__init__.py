"""Inference adapters (spec sections 18, 29, 30, 44, 45).

Three concerns live here, each behind the frozen ``LLMProvider`` port:

*   ``openai_compatible`` — HTTP against any OpenAI-dialect server (vLLM first);
*   ``fake`` — a deterministic provider for CI and GPU-less development;
*   ``structured`` / ``prompts`` — the schema validation and the versioned
    templates that turn a model into something an orchestrator can trust.
"""

from __future__ import annotations

from infrastructure.llm.errors import PromptNotFoundError, PromptRenderError
from infrastructure.llm.fake import FakeLLMProvider, FakeLLMProviderFactory, ScriptedResponse
from infrastructure.llm.openai_compatible import (
    HttpLLMProviderFactory,
    OpenAICompatibleLLMProvider,
    OpenAICompatibleSettings,
)
from infrastructure.llm.prompts import PromptLibrary, PromptTemplate, RenderedPrompt
from infrastructure.llm.structured import (
    OUTPUT_MODELS,
    CoderOutput,
    PlannedTask,
    PlannerOutput,
    ReviewerFinding,
    ReviewerOutput,
    StructuredCompletion,
    StructuredOutputParser,
)

__all__ = [
    "OUTPUT_MODELS",
    "CoderOutput",
    "FakeLLMProvider",
    "FakeLLMProviderFactory",
    "HttpLLMProviderFactory",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleSettings",
    "PlannedTask",
    "PlannerOutput",
    "PromptLibrary",
    "PromptNotFoundError",
    "PromptRenderError",
    "PromptTemplate",
    "RenderedPrompt",
    "ReviewerFinding",
    "ReviewerOutput",
    "ScriptedResponse",
    "StructuredCompletion",
    "StructuredOutputParser",
]
