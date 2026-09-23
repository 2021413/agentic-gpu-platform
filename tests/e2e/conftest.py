"""End-to-end fixtures: the real platform, minus the GPU.

Real PostgreSQL, real mappers, real orchestrator, real git worktrees, real
sandboxed tools. Only the model is fake — which is the point of the spec's
acceptance criterion: the whole workflow must work without a GPU.
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from bootstrap.config import Environment, LLMProviderKind, Settings
from bootstrap.container import Container, build_container
from domain.entities.project import ToolchainConfig
from infrastructure.llm import FakeLLMProviderFactory

pytestmark = pytest.mark.e2e

# /bin/true and /bin/false are the only build and test commands that behave
# identically on every machine, which is what a deterministic e2e needs.
PASSING_COMMAND = "/bin/true"
FAILING_COMMAND = "/bin/false"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=e2e@example.invalid", "-c", "user.name=e2e", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def sample_repository(tmp_path: Path) -> Path:
    """A real git repository for the agents to work on."""
    repo = tmp_path / "sample-project"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "parser.py").write_text("def parse(payload: bytes) -> int:\n    return 0\n")
    (repo / "README.md").write_text("# sample\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def toolchain() -> ToolchainConfig:
    return ToolchainConfig(
        language="python", build_command=PASSING_COMMAND, test_command=PASSING_COMMAND
    )


@pytest.fixture
def e2e_settings(postgres_url: str, tmp_path: Path) -> Settings:
    return Settings(
        # `Settings` reads `.env`, so without this the suite inherits whatever
        # the developer happens to be pointing at. Setting REQUIRE_APPROVAL=true
        # locally to guard a real repository made an end-to-end test assert
        # COMPLETED against a run correctly parked in AWAITING_APPROVAL — a test
        # that fails for a reason outside the repository is not a test.
        _env_file=None,  # type: ignore[call-arg]
        environment=Environment.CI,
        database_url=postgres_url,
        service_token="e2e-token",
        llm_provider=LLMProviderKind.FAKE,
        # Stated rather than inherited: this suite asserts that a run reaches
        # COMPLETED on its own, which is only true when nothing holds it.
        require_approval=False,
        # Uploaded projects land here rather than in /projects, which does not
        # exist on a developer machine and must not on a CI runner.
        projects_root=tmp_path / "projects",
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        prompts_root=Path("prompts"),
        job_lease_seconds=60.0,
        heartbeat_interval_seconds=1.0,
        heartbeat_timeout_seconds=30.0,
        max_parallel_candidates=3,
    )


@pytest.fixture
async def container(e2e_settings: Settings) -> AsyncIterator[Container]:
    """The real container, with messaging in process and no GPU behind it."""
    built = await build_container(
        e2e_settings,
        in_memory_messaging=True,
        llm_factory=FakeLLMProviderFactory(
            model_id=e2e_settings.model_id,
            context_length=e2e_settings.model_context_length,
        ),
    )
    try:
        yield built
    finally:
        await built.aclose()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
