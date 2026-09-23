"""The worker process assembles itself correctly (spec section 45).

The compose stack ships a CPU-only worker backed by the deterministic fake
provider. It could not register: `build_agent` always built an HTTP probe
pointed at an inference server, and with the fake provider there is no server
to point at — nothing listens, so the agent waited out its 900-second startup
timeout and the pool stayed empty. A run then failed with "no compatible
worker is available", which named a symptom three layers from the cause.

The fake provider runs inside the control plane; a worker backed by it has
nothing to poll and must not pretend otherwise.
"""

from __future__ import annotations

import asyncio

import pytest

from bootstrap.config import LLMProviderKind, WorkerSettings
from worker_agent.main import build_agent


def settings(provider: LLMProviderKind) -> WorkerSettings:
    return WorkerSettings(
        control_plane_url="http://api:8000",
        worker_endpoint="http://worker-fake:9000",
        # Deliberately unreachable: nothing must ever connect to it.
        inference_base_url="http://127.0.0.1:1",
        llm_provider=provider,
        model_id="fake-coder-1",
        _env_file=None,  # type: ignore[call-arg]
    )


def test_the_fake_worker_does_not_wait_for_a_server_that_does_not_exist() -> None:
    agent, _, probe = build_agent(settings(LLMProviderKind.FAKE))

    async def ready_quickly() -> bool:
        # A tenth of a second against a 900-second default: if this probe
        # touches the network at all, it cannot answer inside the timeout.
        async with asyncio.timeout(0.1):
            return await probe.wait_until_ready(timeout_seconds=0.05)

    assert asyncio.run(ready_quickly()) is True
    assert asyncio.run(probe.is_healthy()) is True
    assert agent is not None


def test_the_fake_worker_declares_no_context_length_it_cannot_know() -> None:
    """It serves nothing, so it has no served window to report. The declared
    value must stand rather than be replaced by a guess."""
    _, _, probe = build_agent(settings(LLMProviderKind.FAKE))

    assert asyncio.run(probe.served_context_length()) is None


def test_a_real_worker_still_waits_for_its_engine() -> None:
    """The narrow fix must not disarm the guard it sits next to: registering
    before the engine can serve advertises capacity that does not exist."""
    _, _, probe = build_agent(settings(LLMProviderKind.OPENAI_COMPATIBLE))

    assert asyncio.run(probe.is_healthy()) is False


@pytest.mark.parametrize("provider", list(LLMProviderKind))
def test_every_provider_kind_assembles(provider: LLMProviderKind) -> None:
    """A new provider must not silently fall through to a probe that hangs."""
    agent, client, probe = build_agent(settings(provider))

    assert agent is not None and client is not None and probe is not None


# -- serverless engines ----------------------------------------------------
#
# A Modal Server has no container when it is idle, and its proxy starts one in
# response to the request that finds the pool empty. That makes a liveness probe
# an *expensive* operation: the heartbeat runs every ten seconds, so probing a
# scale-to-zero endpoint on that schedule either keeps an H100 alive around the
# clock or pays a cold start six times a minute. Both defeat the reason for
# moving to it.


def serverless_settings() -> WorkerSettings:
    return WorkerSettings(
        control_plane_url="http://api:8000",
        worker_endpoint="https://workspace--worker.modal.direct",
        # Unreachable on purpose: any request at all is the bug under test.
        inference_base_url="http://127.0.0.1:1",
        llm_provider=LLMProviderKind.OPENAI_COMPATIBLE,
        inference_scale_to_zero=True,
        model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        _env_file=None,  # type: ignore[call-arg]
    )


def test_a_serverless_worker_does_not_wake_a_gpu_to_say_it_is_alive() -> None:
    """`is_healthy` runs on every heartbeat. Against Modal each call would boot
    a container, so the answer has to come without touching the network."""
    _, _, probe = build_agent(serverless_settings())

    async def quickly() -> bool:
        # Two hundredths of a second against an unroutable address: a probe
        # that opened a socket could not answer inside this budget.
        async with asyncio.timeout(0.02):
            return await probe.is_healthy()

    assert asyncio.run(quickly()) is True


def test_a_dedicated_worker_still_asks_its_engine() -> None:
    """The exemption must be exactly as wide as the reason for it: an engine
    that died on a dedicated pod must still stop receiving work."""
    _, _, probe = build_agent(settings(LLMProviderKind.OPENAI_COMPATIBLE))

    assert asyncio.run(probe.is_healthy()) is False


def test_a_serverless_worker_still_probes_once_before_registering() -> None:
    """Registration is the only chance to ask the engine what it really serves,
    and that reconciliation has already caught a wrong context length and a
    wrong model name. One cold start at startup buys it; one per heartbeat does
    not, which is the whole distinction."""
    _, _, probe = build_agent(serverless_settings())

    # Unroutable, so a real probe fails — which is the point: it was attempted.
    assert asyncio.run(probe.wait_until_ready(timeout_seconds=0.05)) is False
