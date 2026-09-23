"""The control plane must present the key the engine demands.

The gpu-worker image documents VLLM_API_KEY as the thing standing between an
exposed Pod and the whole internet, and recommends setting it. The adapter can
send a bearer token — `OpenAICompatibleSettings.api_key` exists and is used —
but the composition root never passed one. With the key set on the engine,
every single inference call from the control plane would have come back 401,
on a Pod already billing by the second.

Two halves of a credential that only meet in production, which is the one place
the fake provider cannot reach.
"""

from __future__ import annotations

import pytest

from bootstrap.config import Environment, LLMProviderKind, Settings
from bootstrap.container import _build_llm_factory
from infrastructure.llm.openai_compatible import HttpLLMProviderFactory


def settings(key: str) -> Settings:
    return Settings(
        environment=Environment.CI,
        database_url="postgresql+asyncpg://u:p@localhost/db",
        service_token="t",
        llm_provider=LLMProviderKind.OPENAI_COMPATIBLE,
        inference_api_key=key,
    )


def test_the_engine_key_reaches_the_adapter() -> None:
    factory = _build_llm_factory(settings("sk-vllm-secret"))

    assert isinstance(factory, HttpLLMProviderFactory)
    assert factory._settings.api_key == "sk-vllm-secret"


def test_no_key_configured_sends_no_header() -> None:
    """An engine left open must not receive a literal "Bearer "."""
    factory = _build_llm_factory(settings(""))

    assert isinstance(factory, HttpLLMProviderFactory)
    assert not factory._settings.api_key


def test_the_key_is_not_rendered_by_the_settings_object() -> None:
    """It travels through logs and error reports; it must not be readable there."""
    rendered = repr(settings("sk-vllm-secret"))

    assert "sk-vllm-secret" not in rendered


@pytest.mark.parametrize("key", ["sk-abc", ""])
def test_the_adapter_agrees_with_the_setting(key: str) -> None:
    """Asserted on the header the adapter would actually send, not on the field."""
    factory = _build_llm_factory(settings(key))
    assert isinstance(factory, HttpLLMProviderFactory)
    client = factory._client_for.__self__._default_client("http://gpu:8000")  # type: ignore[attr-defined]

    sent = client.headers.get("authorization")
    assert sent == (f"Bearer {key}" if key else None)
