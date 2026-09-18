"""Structured output extraction, validation and bounded repair (spec section 30).

The behaviour under test is the guarantee the orchestrator relies on: either a
control decision is a validated document, or the job fails loudly. Nothing in
between, and never a decision read out of prose.
"""

from __future__ import annotations

import json

import pytest

from domain.entities.review import Severity
from domain.enums import AgentRole, ReviewVerdict
from domain.exceptions import InferenceError, StructuredOutputError
from domain.value_objects.llm import (
    ChatMessage,
    ChatRole,
    CompletionRequest,
    FinishReason,
    ToolCall,
)
from infrastructure.llm.fake import FakeLLMProvider, ScriptedResponse
from infrastructure.llm.structured import (
    OUTPUT_MODELS,
    CoderOutput,
    PlannerOutput,
    ReviewerOutput,
    StructuredOutputParser,
    extract_json_object,
)

PLAN = {
    "objective": "add retries to the uploader",
    "tasks": [
        {"key": "impl", "title": "implement retries"},
        {"key": "test", "title": "cover retries", "depends_on": ["impl"]},
    ],
}

DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"


def request(text: str = "plan this") -> CompletionRequest:
    return CompletionRequest(messages=(ChatMessage.user(text),))


# -- models ----------------------------------------------------------------
def test_every_role_has_an_output_model() -> None:
    assert set(OUTPUT_MODELS) == set(AgentRole)


def test_the_schema_is_titled_after_the_model() -> None:
    """The title is what identifies the expected document server-side."""
    parser = StructuredOutputParser(PlannerOutput)

    assert parser.json_schema["title"] == "PlannerOutput"
    assert parser.schema_name == "PlannerOutput"
    assert json.loads(parser.schema_text())["title"] == "PlannerOutput"


def test_a_plan_keeps_its_execution_layers() -> None:
    plan = PlannerOutput.model_validate(PLAN)

    assert plan.execution_layers() == (("impl",), ("test",))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({**PLAN, "tasks": []}, "at least 1 item"),
        (
            {
                **PLAN,
                "tasks": [
                    {"key": "a", "title": "A", "depends_on": ["b"]},
                    {"key": "b", "title": "B", "depends_on": ["a"]},
                ],
            },
            "cycle",
        ),
        (
            {**PLAN, "tasks": [{"key": "a", "title": "A", "depends_on": ["ghost"]}]},
            "unknown tasks",
        ),
        (
            {
                **PLAN,
                "tasks": [{"key": "a", "title": "A"}, {"key": "a", "title": "again"}],
            },
            "duplicate",
        ),
        ({**PLAN, "tasks": [{"key": "not a slug", "title": "A"}]}, "slug"),
    ],
)
def test_unusable_plans_are_rejected(payload: dict[str, object], expected: str) -> None:
    parser = StructuredOutputParser(PlannerOutput)

    with pytest.raises(StructuredOutputError, match=expected):
        parser.parse(json.dumps(payload))


def test_a_coder_answer_must_carry_a_real_diff() -> None:
    parser = StructuredOutputParser(CoderOutput)
    prose = {"summary": "done", "diff": "I changed the uploader to retry three times."}

    with pytest.raises(StructuredOutputError, match="unified diff"):
        parser.parse(json.dumps(prose))


def test_a_coder_answer_converts_to_a_domain_patch() -> None:
    output = CoderOutput.model_validate({"summary": "done", "diff": DIFF})

    patch = output.to_patch(base_revision="abc123")

    assert patch.changed_paths == ("a.py",)
    assert patch.total_churn == 2
    assert patch.base_revision == "abc123"


def test_a_failing_review_must_be_actionable() -> None:
    parser = StructuredOutputParser(ReviewerOutput)

    with pytest.raises(StructuredOutputError, match="at least one finding"):
        parser.parse(json.dumps({"verdict": "FAIL", "summary": "nope", "findings": []}))


def test_a_review_converts_to_domain_findings() -> None:
    output = ReviewerOutput.model_validate(
        {
            "verdict": "FAIL",
            "findings": [
                {
                    "summary": "unhandled error path",
                    "severity": "BLOCKER",
                    "file": "a.py",
                    "line": 12,
                    "repair_instruction": "wrap the call and retry",
                }
            ],
        }
    )

    findings = output.to_findings()

    assert output.verdict is ReviewVerdict.FAIL
    assert findings[0].severity is Severity.BLOCKER
    assert findings[0].severity.blocks_completion is True
    assert findings[0].file == "a.py"
    assert findings[0].repair_instruction == "wrap the call and retry"


def test_unknown_fields_do_not_fail_a_valid_answer() -> None:
    parser = StructuredOutputParser(PlannerOutput)

    plan = parser.parse(json.dumps({**PLAN, "confidence": 0.7}))

    assert plan.objective == "add retries to the uploader"


# -- extraction ------------------------------------------------------------
def test_clean_json_is_parsed() -> None:
    assert extract_json_object(json.dumps(PLAN)) == PLAN


def test_json_wrapped_in_prose_and_fences_is_recovered() -> None:
    text = (
        "Sure! Here is the plan you asked for:\n\n"
        "```json\n" + json.dumps(PLAN) + "\n```\n\n"
        "Let me know if you want me to adjust it."
    )

    assert extract_json_object(text) == PLAN


def test_json_wrapped_in_bare_prose_is_recovered() -> None:
    text = "Here it is: " + json.dumps(PLAN) + " Hope that helps!"

    assert extract_json_object(text) == PLAN


def test_braces_inside_strings_do_not_end_the_object() -> None:
    payload = {"summary": "handles {weird} input", "diff": DIFF + '+{"a": 1}\n'}
    text = "Answer:\n" + json.dumps(payload)

    assert extract_json_object(text) == payload


def test_a_draft_inside_a_reasoning_block_is_ignored() -> None:
    """A ``<think>`` block usually holds a *draft* that differs from the answer."""
    draft = json.dumps({"objective": "draft, not the answer", "tasks": []})
    text = f"<think>let me try {draft} ... no, better:</think>\n" + json.dumps(PLAN)

    assert extract_json_object(text) == PLAN


def test_an_unclosed_reasoning_block_does_not_hide_the_answer() -> None:
    text = "some rambling preamble</think>\n" + json.dumps(PLAN)

    assert extract_json_object(text) == PLAN


def test_prose_alone_yields_nothing() -> None:
    assert extract_json_object("I think the plan is fine, honestly.") is None


def test_missing_json_is_reported_as_a_structured_output_error() -> None:
    parser = StructuredOutputParser(PlannerOutput)

    with pytest.raises(StructuredOutputError) as excinfo:
        parser.parse("no json here")

    assert excinfo.value.schema == "PlannerOutput"
    assert excinfo.value.raw_output == "no json here"


# -- repair loop -----------------------------------------------------------
def test_the_repair_prompt_names_the_actual_validation_error() -> None:
    parser = StructuredOutputParser(PlannerOutput)

    prompt = parser.repair_prompt("{}", "- tasks: Field required")

    assert "tasks: Field required" in prompt
    assert "PlannerOutput" in prompt
    assert "JSON only" in prompt


async def test_an_invalid_answer_is_repaired_within_the_budget() -> None:
    parser = StructuredOutputParser(PlannerOutput, max_repair_attempts=2)
    provider = FakeLLMProvider(
        script=[
            ScriptedResponse.invalid_structured_output("Honestly the plan is obvious."),
            ScriptedResponse.text(json.dumps(PLAN)),
        ]
    )

    completion = await parser.complete(provider, request())

    assert completion.attempts == 2
    assert completion.value.objective == "add retries to the uploader"
    # The failed attempt is paid for, so it is accounted for too.
    assert completion.usage.total_tokens > completion.result.usage.total_tokens
    second_call = provider.calls[1]
    assert second_call.messages[-2].role is ChatRole.ASSISTANT
    assert second_call.messages[-1].role is ChatRole.USER
    assert "Validation error" in second_call.messages[-1].content


async def test_exhausting_the_repair_budget_fails_explicitly() -> None:
    parser = StructuredOutputParser(PlannerOutput, max_repair_attempts=1)
    provider = FakeLLMProvider(
        script=[
            ScriptedResponse.invalid_structured_output(),
            ScriptedResponse.invalid_structured_output(),
            ScriptedResponse.text(json.dumps(PLAN)),
        ]
    )

    with pytest.raises(StructuredOutputError) as excinfo:
        await parser.complete(provider, request())

    assert excinfo.value.attempt == 2
    assert excinfo.value.schema == "PlannerOutput"
    assert provider.call_count == 2  # the budget, not one call more


async def test_a_truncated_answer_fails_without_burning_a_repair() -> None:
    """Re-asking with the same token budget would truncate identically."""
    parser = StructuredOutputParser(PlannerOutput, max_repair_attempts=2)
    provider = FakeLLMProvider(script=[ScriptedResponse.truncated()])

    with pytest.raises(StructuredOutputError, match="token limit"):
        await parser.complete(provider, request())

    assert provider.call_count == 1


async def test_inference_failures_are_not_treated_as_model_mistakes() -> None:
    parser = StructuredOutputParser(PlannerOutput)
    provider = FakeLLMProvider(script=[ScriptedResponse.server_error()])

    with pytest.raises(InferenceError):
        await parser.complete(provider, request())

    assert provider.call_count == 1


async def test_the_schema_is_attached_to_the_request() -> None:
    parser = StructuredOutputParser(PlannerOutput)
    provider = FakeLLMProvider(script=[ScriptedResponse.text(json.dumps(PLAN))])

    await parser.complete(provider, request())

    assert provider.calls[0].json_schema == parser.json_schema


async def test_a_caller_supplied_schema_is_left_alone() -> None:
    parser = StructuredOutputParser(PlannerOutput)
    pinned = {"title": "PlannerOutput", "type": "object"}
    provider = FakeLLMProvider(script=[ScriptedResponse.text(json.dumps(PLAN))])

    await parser.complete(
        provider, CompletionRequest(messages=(ChatMessage.user("plan"),), json_schema=pinned)
    )

    assert provider.calls[0].json_schema == pinned


async def test_an_answer_returned_as_a_tool_call_is_still_validated() -> None:
    parser = StructuredOutputParser(PlannerOutput)
    provider = FakeLLMProvider(
        script=[
            ScriptedResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="emit_plan", arguments=json.dumps(PLAN)),),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        ]
    )

    completion = await parser.complete(provider, request())

    assert completion.value.tasks[0].key == "impl"


def test_a_negative_repair_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_repair_attempts"):
        StructuredOutputParser(PlannerOutput, max_repair_attempts=-1)
