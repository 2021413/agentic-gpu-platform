"""Environment-driven configuration (spec section 22).

Settings are read once, validated, and then passed explicitly to whatever needs
them. No module reaches into the environment on its own: a value that can change
behaviour must be visible in one place, and testable by construction.

Names here are the contract with ``.env.example`` and ``docker-compose.yml``.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum, unique
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from domain.enums import AgentRole
from domain.value_objects.limits import RunLimits

__all__ = [
    "ApiSettings",
    "LLMProviderKind",
    "Settings",
    "WorkerSettings",
    "get_settings",
]


@unique
class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION


@unique
class LLMProviderKind(StrEnum):
    """Which inference adapter to wire in.

    ``fake`` is not a test-only convenience: it is how the whole stack runs
    locally and in CI without a GPU, which the spec requires.
    """

    OPENAI_COMPATIBLE = "openai_compatible"
    FAKE = "fake"


@unique
class SchedulerStrategy(StrEnum):
    LEAST_LOADED = "least_loaded_compatible_worker"
    ROUND_ROBIN = "round_robin"


class Settings(BaseSettings):
    """Control-plane configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    environment: Environment = Environment.LOCAL
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True

    # -- stores ---------------------------------------------------------
    database_url: str = "postgresql+asyncpg://agentic:agentic@localhost:5432/agentic"
    redis_url: str = "redis://localhost:6379/0"

    # -- http -----------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    service_token: SecretStr = SecretStr("")

    # -- inference ------------------------------------------------------
    llm_provider: LLMProviderKind = LLMProviderKind.OPENAI_COMPATIBLE
    model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    model_context_length: int = 262_144
    llm_request_timeout_seconds: float = 600.0

    # -- scheduling and jobs --------------------------------------------
    scheduler_strategy: SchedulerStrategy = SchedulerStrategy.LEAST_LOADED
    heartbeat_interval_seconds: float = 10.0
    heartbeat_timeout_seconds: float = 45.0
    job_lease_seconds: float = 120.0
    job_max_attempts: int = 3
    reaper_interval_seconds: float = 10.0
    executor_concurrency: int = 8

    # -- agentic budgets ------------------------------------------------
    max_plan_revisions: int = 2
    max_coder_iterations: int = 6
    max_repair_iterations: int = 3
    max_parallel_candidates: int = 3

    # -- filesystem -----------------------------------------------------
    workspace_root: Path = Path("/var/lib/agentic/workspaces")
    artifact_root: Path = Path("/var/lib/agentic/artifacts")
    prompts_root: Path = Path("prompts")

    # -- validation -----------------------------------------------------
    static_analysis_is_blocking: bool = False

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _check_coherence(self) -> Settings:
        if self.heartbeat_interval_seconds <= 0 or self.heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat interval and timeout must be positive")
        if self.heartbeat_timeout_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "heartbeat_timeout_seconds must exceed heartbeat_interval_seconds, "
                "otherwise a healthy worker is reaped between two beats"
            )
        if self.environment.is_production and not self.service_token.get_secret_value():
            raise ValueError("SERVICE_TOKEN is required outside local development")
        if self.environment.is_production and self.llm_provider is LLMProviderKind.FAKE:
            raise ValueError("the fake inference provider must never run in production")
        return self

    # -- derived --------------------------------------------------------
    @property
    def heartbeat_interval(self) -> timedelta:
        return timedelta(seconds=self.heartbeat_interval_seconds)

    @property
    def heartbeat_timeout(self) -> timedelta:
        return timedelta(seconds=self.heartbeat_timeout_seconds)

    @property
    def heartbeat_ttl(self) -> timedelta:
        """Registry TTL. Slightly longer than the reap timeout so the registry
        never forgets a worker the orchestrator still considers merely late."""
        return timedelta(seconds=self.heartbeat_timeout_seconds * 2)

    @property
    def job_lease(self) -> timedelta:
        return timedelta(seconds=self.job_lease_seconds)

    @property
    def reaper_interval(self) -> timedelta:
        return timedelta(seconds=self.reaper_interval_seconds)

    @property
    def run_limits(self) -> RunLimits:
        return RunLimits(
            max_plan_revisions=self.max_plan_revisions,
            max_coder_iterations=self.max_coder_iterations,
            max_repair_iterations=self.max_repair_iterations,
            max_worker_retries=self.job_max_attempts,
            max_parallel_candidates=self.max_parallel_candidates,
        )


class WorkerSettings(BaseSettings):
    """Configuration of the agent process that lives next to a GPU.

    A worker knows two addresses: where the control plane is, and where its own
    inference server listens. It never knows about the database or the queue.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    control_plane_url: str = "http://localhost:8080"
    service_token: SecretStr = SecretStr("")

    worker_id: str | None = None
    worker_endpoint: str = "http://localhost:8000"
    worker_port: int = 8001
    inference_base_url: str = "http://127.0.0.1:8000"
    inference_api_key: SecretStr = SecretStr("")

    model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    model_context_length: int = 262_144
    worker_concurrency: int = 4
    worker_roles: str = "PLANNER,CODER,REVIEWER"

    heartbeat_interval_seconds: float = 10.0
    worker_drain_timeout_seconds: float = 300.0

    gpu_type: str | None = None
    gpu_count: int = 1
    tensor_parallel_size: int = 1

    log_level: str = "INFO"

    @property
    def roles(self) -> frozenset[AgentRole]:
        """Roles this worker accepts. An unknown name is a configuration error,
        not something to silently drop: a typo would quietly shrink the pool."""
        parsed: set[AgentRole] = set()
        for raw in self.worker_roles.split(","):
            name = raw.strip().upper()
            if not name:
                continue
            try:
                parsed.add(AgentRole(name))
            except ValueError as exc:
                raise ValueError(f"unknown agent role in WORKER_ROLES: {raw!r}") from exc
        if not parsed:
            raise ValueError("WORKER_ROLES must list at least one role")
        return frozenset(parsed)

    @property
    def heartbeat_interval(self) -> timedelta:
        return timedelta(seconds=self.heartbeat_interval_seconds)

    @property
    def drain_timeout(self) -> timedelta:
        return timedelta(seconds=self.worker_drain_timeout_seconds)


class ApiSettings(BaseSettings):
    """Presentation-only knobs, kept apart so the API can be tuned alone."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    cors_allow_origins: str = ""
    sse_keepalive_seconds: float = 15.0
    max_events_backfill: int = 500

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        return tuple(o.strip() for o in self.cors_allow_origins.split(",") if o.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once.

    Cached rather than global-mutable: tests clear the cache instead of
    reaching into a module-level variable.
    """
    return Settings()
