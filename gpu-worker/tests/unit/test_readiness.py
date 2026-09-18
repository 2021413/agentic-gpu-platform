"""Readiness polling and the smoke test, over a simulated vLLM."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx
from tests.conftest import FakeClock, Recorder

from worker.config import Secret, WorkerConfig
from worker.readiness import (
    SMOKE_MAX_TOKENS,
    SMOKE_PROMPT,
    NotReadyError,
    SmokeResult,
    health,
    list_models,
    smoke_test,
    wait_until_ready,
)

BASE_URL = "http://127.0.0.1:8000"
MODEL = "acme/tiny-model"

MODELS_OK = {
    "object": "list",
    "data": [{"id": MODEL, "object": "model", "owned_by": "vllm"}],
}


def _completion(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "int add(int a, int b){return a+b;}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 17, "total_tokens": 28},
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def router() -> Any:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


# ----------------------------------------------------------------------
# list_models
# ----------------------------------------------------------------------


def test_a_well_formed_200_lists_the_models(config: WorkerConfig, router: Any) -> None:
    router.get("/v1/models").respond(200, json=MODELS_OK)

    assert list_models(config) == (MODEL,)


def test_entries_that_are_not_objects_are_ignored(config: WorkerConfig, router: Any) -> None:
    router.get("/v1/models").respond(
        200, json={"object": "list", "data": [{"id": MODEL}, "junk", None, {"no_id": 1}]}
    )

    assert list_models(config) == (MODEL, "")


@pytest.mark.parametrize("status", [404, 500, 503])
def test_a_non_200_is_not_readiness(config: WorkerConfig, router: Any, status: int) -> None:
    router.get("/v1/models").respond(status, json={"detail": "loading"})

    with pytest.raises(NotReadyError) as caught:
        list_models(config)

    assert f"HTTP {status}" in str(caught.value)


def test_invalid_json_is_not_readiness(config: WorkerConfig, router: Any) -> None:
    router.get("/v1/models").respond(
        200, content=b"{not json", headers={"content-type": "application/json"}
    )

    with pytest.raises(NotReadyError) as caught:
        list_models(config)

    assert "did not return JSON" in str(caught.value)


def test_an_html_error_page_from_a_proxy_is_not_readiness(
    config: WorkerConfig, router: Any
) -> None:
    router.get("/v1/models").respond(200, html="<html><body><h1>502 Bad Gateway</h1></body></html>")

    with pytest.raises(NotReadyError) as caught:
        list_models(config)

    assert "did not return JSON" in str(caught.value)


def test_a_json_object_without_a_model_list_is_not_readiness(
    config: WorkerConfig, router: Any
) -> None:
    router.get("/v1/models").respond(200, json={"object": "list", "data": "everything"})

    with pytest.raises(NotReadyError) as caught:
        list_models(config)

    assert "no model list" in str(caught.value)


def test_readiness_uses_the_configured_base_url(make_config: Callable[..., WorkerConfig]) -> None:
    config = make_config(host="0.0.0.0", port=9123)  # noqa: S104
    with respx.mock(base_url="http://127.0.0.1:9123", assert_all_called=True) as mock:
        mock.get("/v1/models").respond(200, json=MODELS_OK)

        assert list_models(config) == (MODEL,)


# ----------------------------------------------------------------------
# the API key travels in a header, or not at all
# ----------------------------------------------------------------------


def test_the_api_key_is_sent_as_a_bearer_header(
    make_config: Callable[..., WorkerConfig], router: Any
) -> None:
    config = make_config(vllm_api_key=Secret("sk-live-1234"))
    route = router.get("/v1/models").respond(200, json=MODELS_OK)

    list_models(config)

    assert route.calls.last.request.headers["authorization"] == "Bearer sk-live-1234"


def test_no_authorization_header_without_a_key(config: WorkerConfig, router: Any) -> None:
    route = router.get("/v1/models").respond(200, json=MODELS_OK)

    list_models(config)

    assert "authorization" not in route.calls.last.request.headers


def test_the_smoke_test_and_health_probe_carry_the_key_too(
    make_config: Callable[..., WorkerConfig], router: Any
) -> None:
    config = make_config(vllm_api_key=Secret("sk-live-1234"))
    completions = router.post("/v1/chat/completions").respond(200, json=_completion())
    healthcheck = router.get("/health").respond(200, text="ok")

    smoke_test(config)
    health(config)

    assert completions.calls.last.request.headers["authorization"] == "Bearer sk-live-1234"
    assert healthcheck.calls.last.request.headers["authorization"] == "Bearer sk-live-1234"


# ----------------------------------------------------------------------
# wait_until_ready
# ----------------------------------------------------------------------


def test_ready_on_the_first_poll(
    config: WorkerConfig, router: Any, clock: FakeClock, recorder: Recorder
) -> None:
    router.get("/v1/models").respond(200, json=MODELS_OK)

    result = wait_until_ready(config, now=clock.now, sleep=clock.sleep, log=recorder)

    assert result.ready is True
    assert result.models == (MODEL,)
    assert result.waited_seconds == 0.0
    assert clock.sleeps == []
    assert recorder.lines == []
    assert "ready after 0s" in result.render()


def test_readiness_arrives_after_a_few_failed_polls(
    config: WorkerConfig, router: Any, clock: FakeClock, recorder: Recorder
) -> None:
    router.get("/v1/models").mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(503, json={"detail": "loading model weights"}),
            httpx.Response(200, json=MODELS_OK),
        ]
    )

    result = wait_until_ready(
        config,
        timeout_seconds=60.0,
        poll_seconds=3.0,
        now=clock.now,
        sleep=clock.sleep,
        log=recorder,
    )

    assert result.ready is True
    assert clock.sleeps == [3.0, 3.0]
    assert result.waited_seconds == 6.0
    assert "not ready yet" in recorder.text


def test_a_served_model_name_is_what_readiness_looks_for(
    make_config: Callable[..., WorkerConfig], router: Any, clock: FakeClock
) -> None:
    config = make_config(served_model_name="acme-public")
    router.get("/v1/models").respond(200, json={"object": "list", "data": [{"id": "acme-public"}]})

    assert wait_until_ready(config, now=clock.now, sleep=clock.sleep).ready is True


def test_a_server_serving_something_else_is_never_ready_and_says_what_it_serves(
    config: WorkerConfig, router: Any, clock: FakeClock, recorder: Recorder
) -> None:
    router.get("/v1/models").respond(
        200,
        json={
            "object": "list",
            "data": [{"id": "meta-llama/Llama-3-8B"}, {"id": "stale/snapshot"}],
        },
    )

    result = wait_until_ready(
        config,
        timeout_seconds=10.0,
        poll_seconds=3.0,
        now=clock.now,
        sleep=clock.sleep,
        log=recorder,
    )

    assert result.ready is False
    assert result.detail is not None
    assert "meta-llama/Llama-3-8B" in result.detail
    assert "stale/snapshot" in result.detail
    assert repr(MODEL) in result.detail
    assert "not ready after" in result.render()


def test_the_deadline_is_respected_to_the_second(
    config: WorkerConfig, router: Any, clock: FakeClock, recorder: Recorder
) -> None:
    router.get("/v1/models").respond(503, json={"detail": "loading"})

    result = wait_until_ready(
        config,
        timeout_seconds=10.0,
        poll_seconds=3.0,
        now=clock.now,
        sleep=clock.sleep,
        log=recorder,
    )

    assert result.ready is False
    # Three full intervals and a final short one that lands exactly on the
    # deadline: the poll never overshoots the budget it was given.
    assert clock.sleeps == [3.0, 3.0, 3.0, 1.0]
    assert result.waited_seconds == 10.0
    assert sum(clock.sleeps) == 10.0
    assert result.detail is not None
    assert "HTTP 503" in result.detail


def test_a_connection_that_never_answers_is_reported_by_type(
    config: WorkerConfig, router: Any, clock: FakeClock
) -> None:
    router.get("/v1/models").mock(side_effect=httpx.ConnectError("connection refused"))

    result = wait_until_ready(
        config, timeout_seconds=3.0, poll_seconds=3.0, now=clock.now, sleep=clock.sleep
    )

    assert result.ready is False
    assert result.detail is not None
    assert "ConnectError" in result.detail
    assert result.models == ()


def test_a_zero_timeout_means_check_once(
    config: WorkerConfig, router: Any, clock: FakeClock
) -> None:
    router.get("/v1/models").respond(503, json={"detail": "loading"})

    result = wait_until_ready(
        config, timeout_seconds=0.0, poll_seconds=3.0, now=clock.now, sleep=clock.sleep
    )

    assert result.ready is False
    assert clock.sleeps == []


# ----------------------------------------------------------------------
# smoke_test
# ----------------------------------------------------------------------


def test_a_correct_completion_passes(config: WorkerConfig, router: Any, clock: FakeClock) -> None:
    route = router.post("/v1/chat/completions").respond(200, json=_completion())

    result = smoke_test(config, now=clock.now)

    assert result.passed is True
    assert result.model == MODEL
    assert result.output == "int add(int a, int b){return a+b;}"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 17
    assert result.detail is None
    assert "smoke test passed" in result.render()[0]

    sent = route.calls.last.request
    body = json.loads(sent.content)
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": SMOKE_PROMPT}]
    assert body["max_tokens"] == SMOKE_MAX_TOKENS
    assert body["temperature"] == 0.0


def test_the_prompt_and_budget_are_overridable(config: WorkerConfig, router: Any) -> None:
    route = router.post("/v1/chat/completions").respond(200, json=_completion())

    smoke_test(config, max_tokens=8, prompt="ping")

    body = json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 8
    assert body["messages"][0]["content"] == "ping"


@pytest.mark.parametrize("status", [401, 404, 500, 503])
def test_a_non_200_fails_the_smoke_test(config: WorkerConfig, router: Any, status: int) -> None:
    router.post("/v1/chat/completions").respond(status, text="upstream exploded")

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert f"HTTP {status}" in result.detail
    assert "upstream exploded" in result.detail
    assert "smoke test FAILED" in result.render()[0]


def test_a_200_that_is_not_json_fails(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(200, html="<h1>hello</h1>")

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert "not JSON" in result.detail


def test_a_json_scalar_instead_of_an_object_fails(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(200, json="ok")

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail == "response was not a JSON object"


def test_an_unexpected_object_kind_fails(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(200, json=_completion(object="text_completion"))

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert "unexpected object 'text_completion'" in result.detail


def test_answering_as_a_different_model_fails(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(
        200, json=_completion(model="meta-llama/Llama-3-8B")
    )

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert "answered as 'meta-llama/Llama-3-8B'" in result.detail
    assert repr(MODEL) in result.detail


def test_an_empty_choices_list_fails(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(200, json=_completion(choices=[]))

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert "carried no choices" in result.detail


def test_a_choice_without_a_message_fails(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(
        200, json=_completion(choices=[{"index": 0, "finish_reason": "stop"}])
    )

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert "no message" in result.detail


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("empty-string", ""),
        ("whitespace", "   \n\t  "),
        ("null", None),
        ("wrong-type", 42),
    ],
)
def test_a_200_that_produced_nothing_never_passes(
    config: WorkerConfig, router: Any, label: str, content: object
) -> None:
    """A worker that answers 'nothing' with a 200 is broken, not healthy."""
    router.post("/v1/chat/completions").respond(
        200,
        json=_completion(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        ),
    )

    result = smoke_test(config)

    assert result.passed is False, label
    assert result.detail is not None
    assert "produced no output" in result.detail


def test_several_problems_are_reported_together(config: WorkerConfig, router: Any) -> None:
    router.post("/v1/chat/completions").respond(
        200,
        json=_completion(
            object="list",
            model="other",
            choices=[{"index": 0, "message": {"role": "assistant", "content": ""}}],
        ),
    )

    result = smoke_test(config)

    assert result.passed is False
    assert result.detail is not None
    assert result.detail.count(";") == 2


def test_a_transport_failure_is_reported_not_raised(
    config: WorkerConfig, router: Any, clock: FakeClock
) -> None:
    router.post("/v1/chat/completions").mock(side_effect=httpx.ConnectError("connection refused"))

    result = smoke_test(config, now=clock.now)

    assert result.passed is False
    assert result.detail is not None
    assert "request failed" in result.detail


def test_a_failed_smoke_result_renders_one_line(config: WorkerConfig) -> None:
    assert SmokeResult(False, detail="nope").render() == ["smoke test FAILED: nope"]


# ----------------------------------------------------------------------
# health
# ----------------------------------------------------------------------


def test_health_is_alive_on_200(config: WorkerConfig, router: Any) -> None:
    router.get("/health").respond(200, text="")

    summary = health(config)

    assert summary.alive is True
    assert summary.detail == "healthy"


@pytest.mark.parametrize("status", [404, 500, 503])
def test_health_is_not_alive_on_anything_else(
    config: WorkerConfig, router: Any, status: int
) -> None:
    router.get("/health").respond(status)

    summary = health(config)

    assert summary.alive is False
    assert f"HTTP {status}" in summary.detail


def test_health_reports_a_refused_connection_without_raising(
    config: WorkerConfig, router: Any
) -> None:
    router.get("/health").mock(side_effect=httpx.ConnectError("connection refused"))

    summary = health(config)

    assert summary.alive is False
    assert "ConnectError" in summary.detail
    assert list(summary.models) == []


def test_not_ready_error_renders_its_hint() -> None:
    error = NotReadyError("vLLM never listed the model", hint="check the container logs")

    rendered = error.render()
    assert "readiness check failed: vLLM never listed the model" in rendered
    assert "fix: check the container logs" in rendered
    assert NotReadyError("bare").render() == "readiness check failed: bare"
