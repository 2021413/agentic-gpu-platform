"""Shared fixtures.

Two rules shape everything here:

* **No ambient state.** Every test starts from an environment scrubbed of the
  variables the worker reads, so a developer with ``HF_TOKEN`` exported does not
  get a different result from CI.
* **No real clock, no real network, no real GPU.** The production code already
  accepts ``now``/``sleep``/``log``/``downloader`` as parameters; these fixtures
  supply deterministic versions rather than patching the standard library.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from worker.config import PersistentLayout, WorkerConfig

# Every variable ``WorkerConfig.from_env`` and the CLI look at. Scrubbed before
# each test so the developer's own shell cannot change an assertion.
WORKER_ENV_VARS = (
    "MODEL_ID",
    "MODEL_REVISION",
    "SERVED_MODEL_NAME",
    "PERSISTENT_ROOT",
    "MIN_FREE_DISK_GB",
    "HOST",
    "PORT",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "TENSOR_PARALLEL_SIZE",
    "AUTO_TENSOR_PARALLEL",
    "VLLM_EXTRA_ARGS",
    "READINESS_TIMEOUT_SECONDS",
    "READINESS_POLL_SECONDS",
    "MODEL_DOWNLOAD_MAX_ATTEMPTS",
    "MODEL_DOWNLOAD_BACKOFF_SECONDS",
    "MODEL_LOCK_TIMEOUT_SECONDS",
    "HF_HUB_DISABLE_XET",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "VLLM_API_KEY",
    "VLLM_IMAGE_TAG",
    "WORKER_ALLOW_EPHEMERAL_STORAGE",
)

RUNNING_AS_ROOT = os.geteuid() == 0
requires_unprivileged = pytest.mark.skipif(
    RUNNING_AS_ROOT,
    reason="root bypasses directory permissions, so an unwritable volume cannot be simulated",
)


@pytest.fixture(autouse=True)
def scrubbed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every worker variable from the ambient environment."""
    for name in WORKER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeClock:
    """A monotonic clock that only moves when the code under test sleeps.

    Injected through the ``now``/``sleep`` parameters the production functions
    already expose, which keeps the tests instantaneous *and* lets them assert
    on the backoff schedule instead of guessing at wall-clock timing.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.time = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.time += seconds

    def advance(self, seconds: float) -> None:
        self.time += seconds


class Recorder:
    """A ``log`` sink that remembers what it was told."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        self.lines.append(message)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    """A directory standing in for the mounted RunPod volume."""
    root = tmp_path / "runpod-volume"
    root.mkdir()
    return root


@pytest.fixture
def make_config(volume: Path) -> Callable[..., WorkerConfig]:
    """Build a config rooted on the fake volume, with retries made cheap."""

    def build(**overrides: object) -> WorkerConfig:
        defaults: dict[str, object] = {
            "model_id": "acme/tiny-model",
            "persistent_root": volume,
            "min_free_disk_gb": 0.0,
            "download_max_attempts": 3,
            "download_backoff_seconds": 5.0,
            "lock_timeout_seconds": 0.05,
            "readiness_poll_seconds": 3.0,
            "readiness_timeout_seconds": 30.0,
        }
        defaults.update(overrides)
        return WorkerConfig(**defaults)  # type: ignore[arg-type]

    return build


@pytest.fixture
def config(make_config: Callable[..., WorkerConfig]) -> WorkerConfig:
    return make_config()


@pytest.fixture
def layout(config: WorkerConfig) -> PersistentLayout:
    layout = config.layout
    for directory in layout.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


@pytest.fixture
def fake_snapshot(volume: Path) -> Iterator[Path]:
    """A snapshot directory laid out the way the hub client leaves one."""
    snapshot = volume / "huggingface" / "hub" / "snapshots" / "deadbeefcafe0123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 2048)
    yield snapshot
