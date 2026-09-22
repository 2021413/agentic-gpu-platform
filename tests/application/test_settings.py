"""Configuration must fail loudly on a combination that cannot work."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bootstrap.config import Environment, LLMProviderKind, Settings, WorkerSettings
from domain.enums import AgentRole


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"environment": Environment.LOCAL, "service_token": "t"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_a_timeout_shorter_than_the_beat_is_rejected() -> None:
    """Otherwise a perfectly healthy worker is reaped between two heartbeats."""
    with pytest.raises(ValidationError, match="must exceed"):
        settings(heartbeat_interval_seconds=30.0, heartbeat_timeout_seconds=10.0)


def test_production_requires_a_service_token() -> None:
    with pytest.raises(ValidationError, match="SERVICE_TOKEN"):
        settings(environment=Environment.PRODUCTION, service_token="")


def test_the_fake_provider_is_refused_in_production() -> None:
    with pytest.raises(ValidationError, match="fake inference provider"):
        settings(environment=Environment.PRODUCTION, llm_provider=LLMProviderKind.FAKE)


def test_run_limits_are_derived_from_configuration() -> None:
    limits = settings(max_repair_iterations=7, max_parallel_candidates=2).run_limits
    assert limits.max_repair_iterations == 7
    assert limits.max_parallel_candidates == 2


def test_the_registry_ttl_outlives_the_reap_timeout() -> None:
    """The registry must not forget a worker the orchestrator thinks is merely late."""
    config = settings(heartbeat_timeout_seconds=45.0)
    assert config.heartbeat_ttl > config.heartbeat_timeout


def test_worker_roles_are_parsed_and_typos_are_fatal() -> None:
    assert WorkerSettings(worker_roles="planner, coder").roles == frozenset(
        {AgentRole.PLANNER, AgentRole.CODER}
    )
    with pytest.raises(ValueError, match="unknown agent role"):
        _ = WorkerSettings(worker_roles="PLANNER,CODR").roles
    with pytest.raises(ValueError, match="at least one role"):
        _ = WorkerSettings(worker_roles=" , ").roles


def test_the_service_token_is_not_printed_by_accident() -> None:
    assert "super-secret" not in repr(settings(service_token="super-secret"))


# -- the prompt is more than the code excerpt ------------------------------
def test_the_prompt_overhead_allowance_reaches_the_orchestrator() -> None:
    """The fleet reports room for a whole prompt; the excerpt is only part of
    it. Counting only the excerpt made the first real run fail by exactly one
    token — 28672 packed into a 32768 window with 4096 reserved for the answer,
    and the template pushed it to 28673."""
    settings = Settings(prompt_overhead_tokens=3_000, _env_file=None)  # type: ignore[call-arg]

    assert settings.prompt_overhead_tokens == 3_000
