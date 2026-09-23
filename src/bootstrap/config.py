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
    "PUBLISHED_DEV_SERVICE_TOKEN",
    "ApiSettings",
    "LLMProviderKind",
    "Settings",
    "WorkerSettings",
    "get_api_settings",
    "get_settings",
]


PUBLISHED_DEV_SERVICE_TOKEN = "dev-service-token-change-me"
"""The SERVICE_TOKEN default that ``docker-compose.yml`` and ``.env.example`` set.

It lives here as a name rather than inside the validator because it is not a
magic string: it is the same contract with those two committed files that the
module docstring describes, and the day the compose default changes, this is
the one place that has to follow. The validator below refuses it in production,
and the tests assert against this name instead of retyping the secret.
"""


@unique
class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION

    @property
    def is_local(self) -> bool:
        """Local and CI both run without a GPU and without Docker-in-Docker."""
        return self in (Environment.LOCAL, Environment.CI)


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
    inference_api_key: SecretStr = SecretStr("")
    require_approval: bool = False
    """Hold a reviewed run until a human approves the merge.

    Off by default on purpose: integration writes into someone else's
    repository, but a run waiting on an approval nobody is watching never
    finishes. Turning this on is a statement that somebody is watching.
    """
    """Bearer token the inference engines demand, if they demand one.

    vLLM started with VLLM_API_KEY refuses everything without it. The adapter
    has always been able to send a token; nothing passed one, so an engine
    secured as its own documentation recommends would have answered 401 to
    every call, on hardware billing by the second.

    Empty means the engines are open, which on an exposed Pod means open to
    the internet. An empty value sends no header rather than a literal
    "Bearer ".

    On a Modal Server the value is a proxy token, `wk-<id>.ws-<secret>`. Modal
    accepts the pair joined by a period in exactly this header, which is why
    moving to Modal needs no new authentication mechanism here.
    """

    inference_scale_to_zero: bool = False
    """Whether the inference endpoint is serverless and may have no worker.

    Set it for a Modal Server, leave it off for a RunPod Pod. It changes what
    HTTP 503 means: on a serverless endpoint the platform answers 503 from its
    proxy when the pool is empty and boots a container in response, so the
    status means "the GPU is starting" and the adapter waits. On a dedicated
    Pod the same status means something is wrong, and waiting would hide it.

    Leaving this off against a Modal endpoint produces a specific and expensive
    failure: every first request after an idle period fails, the job is
    requeued, and the retried job runs on the container the failed one paid to
    start — a cold start billed on every cycle and attributed to nothing.
    """

    inference_cold_start_max_wait_seconds: float = 900.0
    """How long a request may wait for a serverless worker to appear.

    Separate from `llm_request_timeout_seconds` because it measures a different
    thing: a container booting, not a model generating. It should cover a cold
    vLLM start — weights resident, CUDA graphs captured — with margin.
    """

    # -- scheduling and jobs --------------------------------------------
    scheduler_strategy: SchedulerStrategy = SchedulerStrategy.LEAST_LOADED
    heartbeat_interval_seconds: float = 10.0
    heartbeat_timeout_seconds: float = 45.0
    job_lease_seconds: float = 120.0
    tool_sandbox_image: str = "python:3.12-slim"
    """Image the deterministic tools run in when the docker sandbox is used.

    It must contain the project's build and test tools. The default carries a
    Python interpreter and nothing else — a C project pointed at it fails every
    build with "make: not found", and every candidate is then non-viable for a
    reason that has nothing to do with the code. Point it at an image that can
    build what you are working on (for example `gcc:13`).

    Known limitation: this is one image for the whole deployment, not one per
    project. A control plane serving projects in different languages needs the
    image on ToolchainConfig instead, which is a schema change.
    """
    reserved_output_tokens: int = 4_096
    """Room kept in every worker's context window for the model's answer.

    Used twice, deliberately from one place: the scheduler subtracts it when
    deciding whether a prompt fits, and it becomes the engine's own max_tokens
    so the generation cannot quietly exceed what was reserved for it. Two
    numbers that had to agree is exactly how the 262144/16384 mismatch
    happened.
    """
    prompt_overhead_tokens: int = 2_048
    """Room kept in the window for the prompt that is not the code excerpt.

    The fleet reports how large a prompt it can take; the repository view is
    only part of that prompt. Instructions, objective, plan, accumulated review
    findings and the JSON schema ride in the same window. Counting only the
    excerpt is what made the first real run fail by exactly one token.
    """
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
        if (
            self.environment.is_production
            and self.service_token.get_secret_value() == PUBLISHED_DEV_SERVICE_TOKEN
        ):
            # A non-empty token passed the check above, which is why this needs its
            # own branch: the compose default is committed to this repository, so a
            # deployment that forgets to change it is guarded by a secret anyone can
            # read. Local development keeps it on purpose, hence the environment test.
            raise ValueError(
                "SERVICE_TOKEN is still the published default value from "
                "docker-compose.yml, which is committed to this repository and so "
                "known to anyone who can read it; set a real secret in production"
            )
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

    inference_scale_to_zero: bool = False
    """Whether this agent's engine is a serverless endpoint that may be at zero.

    The same switch the control plane reads, for a different reason. Here it
    stops the ten-second heartbeat probe from reaching across the network: on a
    Modal Server every such request starts a container, so probing liveness on
    a schedule would keep an H100 warm around the clock. Registration still
    probes once, because reconciling what the engine really serves is worth one
    cold start and has caught real mismatches.
    """

    llm_provider: LLMProviderKind = LLMProviderKind.OPENAI_COMPATIBLE
    """Which provider backs this worker.

    The field has to exist here even though the worker never calls the model
    itself: with the fake provider the inference runs inside the control plane,
    so there is no server at `inference_base_url` to wait for. Without this,
    `extra="ignore"` silently dropped the LLM_PROVIDER the compose file sets,
    the agent waited out its startup timeout against nothing, and the pool
    stayed empty.
    """

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
def get_api_settings() -> ApiSettings:
    """Presentation-only settings, read once like the rest."""
    return ApiSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once.

    Cached rather than global-mutable: tests clear the cache instead of
    reaching into a module-level variable.
    """
    return Settings()
