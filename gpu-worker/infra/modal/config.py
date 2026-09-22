"""Deployment profiles: everything Modal needs to know before a container exists.

These values are read on the machine running `modal deploy`, never inside the
container. That is the whole point of keeping them here: `gpu`, `min_containers`
and `scaledown_window` are arguments to the decorator, so they are frozen into
the deployment and cannot be changed by an environment variable on a Pod.

One profile is selected by `MODAL_PROFILE` (`dev` or `prod`), and every field can
be overridden individually for an experiment without editing this file. The
overrides exist for benchmarking — section 37 of the specification asks for the
same worker deployed at four different `scaledown_window` values — not as a way
to run production off a shell variable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

__all__ = [
    "DEV",
    "PROD",
    "PROFILES",
    "ModalConfigError",
    "ModalWorkerConfig",
    "active_config",
]

# Below this, a cold H100 cannot finish loading 31 GB of FP8 weights and capture
# CUDA graphs before Modal declares the container failed and kills it. Modal's
# own default is 30 seconds, which is correct for a web app and catastrophic
# here: the container dies, the autoscaler starts another, and the loop is
# invisible from the client side except as 503s.
MIN_SANE_STARTUP_TIMEOUT: Final = 300

# Modal bills the GPU for every second a container is alive, idle or not. A long
# scaledown window is a decision to pay for idleness in exchange for fewer cold
# starts; it should be a deliberate one, so an implausible value is refused
# rather than silently accepted.
MAX_SANE_SCALEDOWN_WINDOW: Final = 3600


class ModalConfigError(RuntimeError):
    """A deployment profile that would waste money or expose the worker."""

    def __init__(self, message: str, *, variable: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.variable = variable
        self.hint = hint

    def render(self) -> str:
        parts = [str(self)]
        if self.variable:
            parts.append(f"(set by {self.variable})")
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ModalWorkerConfig:
    """One deployment's worth of Modal Server configuration."""

    profile: str

    # -- hardware -------------------------------------------------------
    gpu: str = "H100"
    """`H100` lets Modal silently substitute an H200 at the same price, which is
    free capacity and free bandwidth for a model that fits either. `H100!`
    forbids the substitution and is what a benchmark must use, because a run
    that sometimes lands on 141 GB of HBM3e is not a measurement of anything."""

    # -- autoscaling ----------------------------------------------------
    min_containers: int = 0
    max_containers: int | None = 1
    buffer_containers: int | None = None
    target_concurrency: float | None = None
    """Left unset means one container handles every request and the pool never
    grows on load alone. That is the right starting point: two repair loops
    sharing one vLLM engine is a throughput question nobody here has measured
    yet, and the specification says to benchmark before enabling it."""

    scaleup_window: int | None = None
    scaledown_window: int = 60
    startup_timeout: int = 900
    exit_grace_period: int = 300
    """A long generation already in flight when the autoscaler decides to scale
    down should finish, not 500. Five minutes is the ceiling on a single
    completion at this context length; beyond that the client has timed out
    anyway."""

    # -- placement ------------------------------------------------------
    routing_region: str = "us-east"
    compute_region: tuple[str, ...] | None = None
    """Deliberately unset. Pinning containers to a region multiplies the GPU
    price and shrinks the pool we are migrating *to* Modal to escape. Set it
    only if network latency to the control plane ever beats GPU availability as
    the binding constraint."""

    cloud: str | None = None

    # -- exposure -------------------------------------------------------
    unauthenticated: bool = False

    # -- cold start -----------------------------------------------------
    enable_memory_snapshot: bool = False
    enable_gpu_snapshot: bool = False
    """Alpha on Modal's side. Phase 3 of the specification, and off by default
    so that the first working deployment does not depend on it."""

    # -- refusals -------------------------------------------------------
    allow_cold_download: bool = False
    """When false, a container that finds no prepared model on the Volume
    refuses to start instead of downloading 31 GB with an H100 on the meter.
    See `scripts/populate_modal_volume.py`."""

    def __post_init__(self) -> None:
        if not self.gpu.strip():
            raise ModalConfigError("gpu must not be empty", variable="MODAL_GPU")
        if self.min_containers < 0:
            raise ModalConfigError(
                "min_containers must not be negative", variable="MODAL_MIN_CONTAINERS"
            )
        if self.max_containers is not None:
            if self.max_containers < 1:
                raise ModalConfigError(
                    "max_containers must be at least 1", variable="MODAL_MAX_CONTAINERS"
                )
            if self.max_containers < self.min_containers:
                raise ModalConfigError(
                    f"max_containers ({self.max_containers}) is below min_containers "
                    f"({self.min_containers})",
                    variable="MODAL_MAX_CONTAINERS",
                )
        if self.target_concurrency is not None and self.target_concurrency < 0:
            raise ModalConfigError(
                "target_concurrency must not be negative", variable="MODAL_TARGET_CONCURRENCY"
            )
        if self.scaledown_window < 1:
            raise ModalConfigError(
                "scaledown_window must be at least 1 second",
                variable="MODAL_SCALEDOWN_WINDOW",
            )
        if self.scaledown_window > MAX_SANE_SCALEDOWN_WINDOW:
            raise ModalConfigError(
                f"scaledown_window of {self.scaledown_window}s keeps an idle GPU alive for "
                f"more than an hour",
                variable="MODAL_SCALEDOWN_WINDOW",
                hint="if a warm worker is genuinely required, set min_containers instead, "
                "where the cost is at least explicit",
            )
        if self.startup_timeout < MIN_SANE_STARTUP_TIMEOUT:
            raise ModalConfigError(
                f"startup_timeout of {self.startup_timeout}s is below the "
                f"{MIN_SANE_STARTUP_TIMEOUT}s a cold vLLM start needs",
                variable="MODAL_STARTUP_TIMEOUT",
                hint="a container killed mid-load is replaced by another that loads from "
                "scratch, and the loop only shows up as 503s at the client",
            )
        if self.enable_gpu_snapshot and not self.enable_memory_snapshot:
            raise ModalConfigError(
                "enable_gpu_snapshot requires enable_memory_snapshot",
                variable="MODAL_ENABLE_GPU_SNAPSHOT",
            )
        if self.unauthenticated and self.profile == "prod":
            raise ModalConfigError(
                "the prod profile refuses unauthenticated=True",
                variable="MODAL_UNAUTHENTICATED",
                hint="an unauthenticated Server URL is an H100 that anyone who finds the "
                "URL can bill to this account; use a proxy token instead",
            )

    # -- derived --------------------------------------------------------
    @property
    def benchmark_gpu(self) -> str:
        """The same GPU request, with Modal's free H200 upgrade refused.

        A cold-start distribution measured across a mixture of H100 and H200 is
        two distributions reported as one.
        """
        return self.gpu if self.gpu.endswith("!") else f"{self.gpu}!"

    def as_server_kwargs(self) -> dict[str, Any]:
        """Exactly the keyword arguments `@app.server()` takes.

        Unset optionals are omitted rather than passed as None, so that Modal's
        own defaults remain visible in the decorator's signature instead of
        being overwritten here with the same value under a different name.
        """
        kwargs: dict[str, Any] = {
            "gpu": self.gpu,
            "min_containers": self.min_containers,
            "scaledown_window": self.scaledown_window,
            "startup_timeout": self.startup_timeout,
            "exit_grace_period": self.exit_grace_period,
            "routing_region": self.routing_region,
            "unauthenticated": self.unauthenticated,
        }
        for name, value in (
            ("max_containers", self.max_containers),
            ("buffer_containers", self.buffer_containers),
            ("target_concurrency", self.target_concurrency),
            ("scaleup_window", self.scaleup_window),
            ("cloud", self.cloud),
        ):
            if value is not None:
                kwargs[name] = value
        if self.compute_region is not None:
            kwargs["compute_region"] = list(self.compute_region)
        if self.enable_memory_snapshot:
            kwargs["enable_memory_snapshot"] = True
        if self.enable_gpu_snapshot:
            kwargs["experimental_options"] = {"enable_gpu_snapshot": True}
        return kwargs

    def describe(self) -> list[str]:
        """What this deployment will cost and refuse, in one block of text."""
        concurrency = (
            "unset (one at a time)"
            if self.target_concurrency is None
            else str(self.target_concurrency)
        )
        regions = "any" if self.compute_region is None else ",".join(self.compute_region)
        auth = "NONE (public URL)" if self.unauthenticated else "proxy token required"
        lines = [
            f"profile              {self.profile}",
            f"gpu                  {self.gpu}",
            f"containers           min={self.min_containers} "
            f"max={self.max_containers if self.max_containers is not None else 'unbounded'}",
            f"target_concurrency   {concurrency}",
            f"scaledown_window     {self.scaledown_window}s",
            f"startup_timeout      {self.startup_timeout}s",
            f"exit_grace_period    {self.exit_grace_period}s",
            f"routing_region       {self.routing_region}",
            f"compute_region       {regions}",
            f"cloud                {self.cloud or 'auto'}",
            f"authentication       {auth}",
            f"memory snapshot      {'on' if self.enable_memory_snapshot else 'off'}"
            f"{' (+GPU, alpha)' if self.enable_gpu_snapshot else ''}",
            f"cold download        {'allowed' if self.allow_cold_download else 'refused'}",
        ]
        if self.min_containers > 0:
            lines.append(
                f"NOTE: min_containers={self.min_containers} bills "
                f"{self.min_containers} GPU(s) continuously, idle or not"
            )
        return lines

    # -- construction ---------------------------------------------------
    def with_overrides(self, environ: Mapping[str, str]) -> ModalWorkerConfig:
        """Apply per-field `MODAL_*` overrides on top of this profile."""
        changes: dict[str, Any] = {}
        for field_name, variable, cast in _OVERRIDES:
            raw = environ.get(variable)
            if raw is None:
                continue
            changes[field_name] = cast(raw, variable)
        return replace(self, **changes) if changes else self


def _integer(raw: str, variable: str) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        raise ModalConfigError(f"expected an integer, got {raw!r}", variable=variable) from None


def _optional_integer(raw: str, variable: str) -> int | None:
    if raw.strip().lower() in ("", "none", "unset", "unbounded"):
        return None
    return _integer(raw, variable)


def _optional_number(raw: str, variable: str) -> float | None:
    text = raw.strip().lower()
    if text in ("", "none", "unset"):
        return None
    try:
        return float(text)
    except ValueError:
        raise ModalConfigError(f"expected a number, got {raw!r}", variable=variable) from None


def _flag(raw: str, variable: str) -> bool:
    text = raw.strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ModalConfigError(f"expected a boolean, got {raw!r}", variable=variable)


def _text(raw: str, variable: str) -> str:
    return raw.strip()


def _optional_text(raw: str, variable: str) -> str | None:
    text = raw.strip()
    return text or None


def _regions(raw: str, variable: str) -> tuple[str, ...] | None:
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or None


_OVERRIDES: Final = (
    ("gpu", "MODAL_GPU", _text),
    ("min_containers", "MODAL_MIN_CONTAINERS", _integer),
    ("max_containers", "MODAL_MAX_CONTAINERS", _optional_integer),
    ("buffer_containers", "MODAL_BUFFER_CONTAINERS", _optional_integer),
    ("target_concurrency", "MODAL_TARGET_CONCURRENCY", _optional_number),
    ("scaleup_window", "MODAL_SCALEUP_WINDOW", _optional_integer),
    ("scaledown_window", "MODAL_SCALEDOWN_WINDOW", _integer),
    ("startup_timeout", "MODAL_STARTUP_TIMEOUT", _integer),
    ("exit_grace_period", "MODAL_EXIT_GRACE_PERIOD", _integer),
    ("routing_region", "MODAL_ROUTING_REGION", _text),
    ("compute_region", "MODAL_COMPUTE_REGION", _regions),
    ("cloud", "MODAL_CLOUD", _optional_text),
    ("unauthenticated", "MODAL_UNAUTHENTICATED", _flag),
    ("enable_memory_snapshot", "MODAL_ENABLE_MEMORY_SNAPSHOT", _flag),
    ("enable_gpu_snapshot", "MODAL_ENABLE_GPU_SNAPSHOT", _flag),
    ("allow_cold_download", "MODAL_ALLOW_COLD_DOWNLOAD", _flag),
)


DEV: Final = ModalWorkerConfig(
    profile="dev",
    gpu="H100",
    min_containers=0,
    # One container, hard. A loop in a script that fans out ten requests would
    # otherwise provision ten H100s, and the mistake is only visible on the bill.
    max_containers=1,
    scaledown_window=60,
    startup_timeout=900,
    exit_grace_period=120,
)

PROD: Final = ModalWorkerConfig(
    profile="prod",
    gpu="H100",
    # Still zero. Section 14: a permanently warm H100 is roughly $2,800 a month,
    # and nothing has yet measured a latency requirement that justifies it.
    min_containers=0,
    max_containers=4,
    scaledown_window=120,
    startup_timeout=900,
    exit_grace_period=300,
)

PROFILES: Final[Mapping[str, ModalWorkerConfig]] = {"dev": DEV, "prod": PROD}


def active_config(environ: Mapping[str, str] | None = None) -> ModalWorkerConfig:
    """The profile this deployment uses, with any overrides applied."""
    env = os.environ if environ is None else environ
    name = env.get("MODAL_PROFILE", "dev").strip().lower()
    try:
        base = PROFILES[name]
    except KeyError:
        raise ModalConfigError(
            f"unknown profile {name!r}",
            variable="MODAL_PROFILE",
            hint=f"known profiles: {', '.join(sorted(PROFILES))}",
        ) from None
    return base.with_overrides(env)
