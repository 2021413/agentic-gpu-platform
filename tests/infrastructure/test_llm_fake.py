"""The deterministic provider used by CI and GPU-less development (spec section 45).

Two properties matter here. The derived answers must be *valid* — a fake that
produced garbage would make every workflow test exercise the repair loop — and
they must be *stable*, so an application test can assert on them. The scripted
mode is the other half: every failure the application layer must handle has to
be reproducible on demand.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from domain.enums import AgentRole, ReviewVerdict
from domain.exceptions import InferenceError, LLMTimeoutError
from domain.ports.llm_provider import LLMProvider, LLMProviderFactory
from domain.value_objects.llm import ChatMessage, CompletionRequest, FinishReason, TokenUsage
from domain.value_objects.worker import WorkerEndpoint
from infrastructure.llm.fake import (
    FakeLLMProvider,
    FakeLLMProviderFactory,
    ScriptedResponse,
)
from infrastructure.llm.structured import (
    CoderOutput,
    PlannerOutput,
    ReviewerOutput,
    StructuredOutputParser,
)

OBJECTIVE = "add retries to the uploader"


def role_request(role: AgentRole, objective: str = OBJECTIVE) -> CompletionRequest:
    """A request shaped like the real ones: role marker plus an objective section."""
    return CompletionRequest(
        messages=(
            ChatMessage.system(f"ROLE: {role.value}\n\nDo your job."),
            ChatMessage.user(f"## Objective\n\n{objective}\n"),
        )
    )


# -- derived answers -------------------------------------------------------
async def test_the_planner_answer_is_a_valid_acyclic_plan() -> None:
    provider = FakeLLMProvider()
    parser = StructuredOutputParser(PlannerOutput)

    completion = await parser.complete(provider, role_request(AgentRole.PLANNER))

    assert completion.attempts == 1
    assert completion.value.objective == OBJECTIVE
    layers = completion.value.execution_layers()
    assert layers[0] == ("analyze",)
    assert set(layers[1]) == {"implement", "cover"}


async def test_the_coder_answer_carries_whole_files() -> None:
    """The double answers in the shape the prompt asks for, not a diff.

    It used to emit a unified diff — the exact form a real model got wrong
    twice in a row, failing a run with "No valid patches in input". A double
    that answers in a form the real one struggles with proves the wrong thing.
    """
    provider = FakeLLMProvider()
    parser = StructuredOutputParser(CoderOutput)

    completion = await parser.complete(provider, role_request(AgentRole.CODER))

    files = completion.value.files
    assert [f.path for f in files] == ["src/add_retries_to_the_uploader.py"]
    assert files[0].content.strip(), "the file was written empty"
    assert "def " in files[0].content


async def test_the_reviewer_answer_passes_by_default() -> None:
    provider = FakeLLMProvider()
    parser = StructuredOutputParser(ReviewerOutput)

    completion = await parser.complete(provider, role_request(AgentRole.REVIEWER))

    assert completion.value.verdict is ReviewVerdict.PASS
    assert completion.value.findings == []


async def test_a_failing_reviewer_can_be_configured() -> None:
    provider = FakeLLMProvider(default_verdict=ReviewVerdict.FAIL)
    parser = StructuredOutputParser(ReviewerOutput)

    completion = await parser.complete(provider, role_request(AgentRole.REVIEWER))

    assert completion.value.verdict is ReviewVerdict.FAIL
    assert completion.value.findings[0].repair_instruction


async def test_answers_are_identical_across_providers_and_calls() -> None:
    first = await FakeLLMProvider().complete(role_request(AgentRole.PLANNER))
    second = await FakeLLMProvider().complete(role_request(AgentRole.PLANNER))
    third = await FakeLLMProvider().complete(role_request(AgentRole.PLANNER))

    assert first.content == second.content == third.content
    assert first.usage == second.usage


async def test_the_answer_follows_the_objective() -> None:
    provider = FakeLLMProvider()

    one = await provider.complete(role_request(AgentRole.CODER, "rename the widget"))
    two = await provider.complete(role_request(AgentRole.CODER, "delete the widget"))

    assert json.loads(one.content)["files_changed"] == ["src/rename_the_widget.py"]
    assert one.content != two.content


async def test_the_objective_is_recovered_from_an_inline_marker() -> None:
    provider = FakeLLMProvider()
    request = CompletionRequest(
        messages=(ChatMessage.user("Objective: ship the parser\nrest of the prompt"),)
    )

    result = await provider.complete(request)

    assert json.loads(result.content)["objective"] == "ship the parser"


async def test_an_unrecognisable_prompt_still_produces_a_valid_answer() -> None:
    provider = FakeLLMProvider()
    parser = StructuredOutputParser(PlannerOutput)

    completion = await parser.complete(
        provider, CompletionRequest(messages=(ChatMessage.user(""),))
    )

    assert completion.value.objective == "unspecified objective"


# -- role inference --------------------------------------------------------
async def test_the_role_comes_from_the_requested_schema_first() -> None:
    provider = FakeLLMProvider(default_role=AgentRole.PLANNER)
    parser = StructuredOutputParser(ReviewerOutput)

    completion = await parser.complete(
        provider,
        CompletionRequest(
            messages=(ChatMessage.user(f"## Objective\n\n{OBJECTIVE}"),),
            json_schema=parser.json_schema,
        ),
    )

    assert isinstance(completion.value, ReviewerOutput)


async def test_the_role_falls_back_to_the_prompt_marker() -> None:
    provider = FakeLLMProvider()

    result = await provider.complete(role_request(AgentRole.CODER))

    assert "files" in json.loads(result.content)


async def test_the_default_role_is_the_last_resort() -> None:
    provider = FakeLLMProvider(default_role=AgentRole.REVIEWER)

    result = await provider.complete(
        CompletionRequest(messages=(ChatMessage.user("just do something"),))
    )

    assert json.loads(result.content)["verdict"] == "PASS"


# -- scripted mode ---------------------------------------------------------
async def test_scripted_answers_are_consumed_in_order_then_derived() -> None:
    provider = FakeLLMProvider().script(
        ScriptedResponse.text("first"), ScriptedResponse.text("second")
    )

    assert (await provider.complete(role_request(AgentRole.PLANNER))).content == "first"
    assert (await provider.complete(role_request(AgentRole.PLANNER))).content == "second"
    assert provider.pending == 0
    derived = await provider.complete(role_request(AgentRole.PLANNER))
    assert json.loads(derived.content)["objective"] == OBJECTIVE
    assert provider.call_count == 3


async def test_scripted_usage_and_finish_reason_are_honoured() -> None:
    provider = FakeLLMProvider(
        script=[
            ScriptedResponse(
                content="partial",
                finish_reason=FinishReason.LENGTH,
                usage=TokenUsage(input_tokens=1_000, output_tokens=2_000),
            )
        ]
    )

    result = await provider.complete(role_request(AgentRole.PLANNER))

    assert result.truncated is True
    assert result.usage.total_tokens == 3_000


async def test_a_scripted_timeout_raises_the_domain_error() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse.timeout(timeout_seconds=12)])

    with pytest.raises(LLMTimeoutError) as excinfo:
        await provider.complete(role_request(AgentRole.PLANNER))

    assert excinfo.value.timeout_seconds == 12


async def test_a_scripted_server_error_raises_the_domain_error() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse.server_error(status_code=503)])

    with pytest.raises(InferenceError) as excinfo:
        await provider.complete(role_request(AgentRole.PLANNER))

    assert excinfo.value.status_code == 503


async def test_slowness_is_observable_and_cancellable() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse.slow(30)])
    task = asyncio.create_task(provider.complete(role_request(AgentRole.PLANNER)))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_slowness_can_breach_a_caller_deadline() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse.slow(30)])

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await provider.complete(role_request(AgentRole.PLANNER))


async def test_invalid_structured_output_drives_the_repair_loop() -> None:
    provider = FakeLLMProvider(
        script=[ScriptedResponse.invalid_structured_output(), ScriptedResponse()]
    )
    parser = StructuredOutputParser(PlannerOutput, max_repair_attempts=1)

    completion = await parser.complete(provider, role_request(AgentRole.PLANNER))

    assert completion.attempts == 2
    assert provider.call_count == 2


# -- streaming and health --------------------------------------------------
async def test_the_fake_stream_is_an_async_generator_too() -> None:
    provider = FakeLLMProvider()

    assert inspect.isasyncgen(provider.stream(role_request(AgentRole.PLANNER)))


async def test_streaming_reassembles_into_the_completion_content() -> None:
    provider = FakeLLMProvider(chunk_size=16)
    expected = (await FakeLLMProvider().complete(role_request(AgentRole.PLANNER))).content

    chunks = [chunk async for chunk in provider.stream(role_request(AgentRole.PLANNER))]

    assert len(chunks) > 1
    assert "".join(chunks) == expected


async def test_streaming_can_be_scripted_chunk_by_chunk() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse(chunks=("one ", "two ", "three"))])

    chunks = [chunk async for chunk in provider.stream(role_request(AgentRole.CODER))]

    assert chunks == ["one ", "two ", "three"]


async def test_streaming_failures_are_scriptable() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse.server_error()])

    stream = provider.stream(role_request(AgentRole.CODER))

    with pytest.raises(InferenceError):
        await anext(stream)


async def test_health_is_controllable() -> None:
    provider = FakeLLMProvider()

    assert await provider.health() is True
    provider.healthy = False
    assert await provider.health() is False


def test_the_fake_reports_the_configured_model() -> None:
    provider = FakeLLMProvider(model_id="ci-model", context_length=1_024)

    assert provider.model_info.model_id == "ci-model"
    assert provider.model_info.context_length == 1_024


def test_reset_clears_script_and_history() -> None:
    provider = FakeLLMProvider(script=[ScriptedResponse.text("x")])

    provider.reset()

    assert provider.pending == 0
    assert provider.calls == ()


# -- factory ---------------------------------------------------------------
def test_the_fake_satisfies_the_ports() -> None:
    factory = FakeLLMProviderFactory()
    provider: LLMProvider = factory.for_endpoint(
        WorkerEndpoint("http://worker:8000"), model_id="ci-model"
    )

    assert isinstance(provider, LLMProvider)
    assert isinstance(factory, LLMProviderFactory)


async def test_a_shared_factory_scripts_a_whole_run_in_one_place() -> None:
    factory = FakeLLMProviderFactory().script(ScriptedResponse.text("planned"))
    first = factory.for_endpoint(WorkerEndpoint("http://worker-a:8000"), model_id="m")
    second = factory.for_endpoint(WorkerEndpoint("http://worker-b:8000"), model_id="m")

    assert first is second
    assert (await first.complete(role_request(AgentRole.PLANNER))).content == "planned"
    assert factory.provider.call_count == 1
    assert factory.requested == [("http://worker-a:8000", "m"), ("http://worker-b:8000", "m")]


def test_an_isolated_factory_gives_each_worker_its_own_fake() -> None:
    factory = FakeLLMProviderFactory(shared=False)

    first = factory.fake_for_endpoint(WorkerEndpoint("http://worker-a:8000"), model_id="m")
    again = factory.fake_for_endpoint(WorkerEndpoint("http://worker-a:8000/"), model_id="m")
    other = factory.fake_for_endpoint(WorkerEndpoint("http://worker-b:8000"), model_id="m")

    assert first is again
    assert first is not other
    assert other.model_info.model_id == "m"
    assert len(factory.providers) == 2
