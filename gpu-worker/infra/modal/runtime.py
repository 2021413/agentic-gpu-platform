"""Container-side logic for the Modal worker.

Everything here runs inside the GPU container, and none of it imports `modal`.
That is on purpose: the decisions this module makes — is the model actually on
the Volume, which compile cache may be reused, when is vLLM genuinely serving —
are the expensive ones, and they should be testable on a laptop with no account,
no GPU and no network.

`app.py` is the Modal wiring. This is the part with opinions.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx

from worker.config import PersistentLayout, WorkerConfig
from worker.gpu import survey_gpus
from worker.model_state import ModelPreparationError, prepare_model, read_marker

__all__ = [
    "LOCAL_SCRATCH",
    "ColdModelError",
    "SleepModeError",
    "StartupRecord",
    "append_startup_record",
    "compile_cache_environment",
    "gpu_slug",
    "launch_vllm",
    "resolve_model",
    "scratch_environment",
    "sleep_vllm",
    "startup_records",
    "wake_vllm",
    "warmup_vllm",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_STARTUP_LOG = "startup.jsonl"

# The container's own disk, not the Volume.
LOCAL_SCRATCH: Final = "/tmp"  # noqa: S108 - a container temp dir, not a shared host path


class ColdModelError(RuntimeError):
    """The Volume holds no usable model and downloading one was not allowed.

    Raised in `@modal.enter()`, which means Modal marks the container failed and
    stops billing within seconds. The alternative — downloading 31.2 GB with an
    H100 attached — costs about forty minutes of GPU time to produce something a
    CPU container could have fetched for a fraction of a cent.
    """


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# -- GPU identity -------------------------------------------------------


def gpu_slug(*, timeout: float = 10.0) -> str:
    """A filesystem-safe name for the GPU this container was given.

    Used to key compiled artifacts. Modal may silently upgrade an `H100`
    request to an H200, so the GPU actually attached — not the one requested —
    is what has to appear in the path. Reusing an H100's CUDA graphs on an H200
    is the kind of mistake that produces a slow worker rather than a failing
    one, which is much worse.
    """
    survey = survey_gpus(timeout=timeout)
    if not survey.gpus:
        return "unknown-gpu"
    name = survey.gpus[0].name.strip().lower()
    slug = _SLUG_RE.sub("-", name).strip("-")
    return slug or "unknown-gpu"


def compile_cache_environment(
    layout: PersistentLayout,
    *,
    gpu: str,
    image_tag: str | None = None,
) -> dict[str, str]:
    """Cache paths for artifacts whose validity depends on the hardware.

    vLLM's `torch.compile` output, the CUDA graphs and the Triton kernels are
    only reusable on the architecture and toolchain that produced them. The
    Hugging Face blobs above them are not: those are the same bytes everywhere,
    which is why they stay at the root of the Volume and these do not.
    """
    tag = image_tag or os.environ.get("VLLM_IMAGE_TAG") or "unknown-image"
    root = layout.root / "compiled" / gpu / _SLUG_RE.sub("-", tag.lower()).strip("-")
    return {
        "VLLM_CACHE_ROOT": str(root / "vllm"),
        "TRITON_CACHE_DIR": str(root / "vllm" / "triton"),
    }


def scratch_environment() -> dict[str, str]:
    """Temporary files belong on the container disk, never on the Volume.

    `PersistentLayout` puts `TMPDIR` on the mounted volume, which is right on
    RunPod: it keeps a download's scratch space off a container disk sized for
    an image. On Modal it is fatal, and not for a reason anyone would guess.

    vLLM talks to its engine core over **ZeroMQ IPC sockets** created under
    `TMPDIR`. A Modal Volume is a FUSE filesystem with no support for Unix
    domain sockets, so `socket.bind()` fails with

        zmq.error.ZMQError: Operation not supported
            (addr='ipc:///data/tmp/...')

    and the engine dies about a minute into loading — after the weights have
    started moving, which is the most expensive place to fail. Observed on the
    first real cold start, not predicted.

    Nothing is lost by moving it: the weights are fetched by the populate job,
    not by this container, so `TMPDIR` here holds sockets and a few small files.
    """
    return {"TMPDIR": LOCAL_SCRATCH}


# -- model availability -------------------------------------------------


def resolve_model(
    config: WorkerConfig,
    *,
    allow_cold_download: bool,
    log: Callable[[str], None] = print,
) -> tuple[Path, bool]:
    """Return the local snapshot to serve, and whether this container fetched it.

    The happy path is a marker written by `scripts/populate_modal_volume.py` and
    verified offline in seconds. Anything else is either an explicit opt-in to
    paying GPU rates for a download, or a refusal that names the command to run.
    """
    marker = read_marker(config.layout)
    if marker is not None and marker.model_id == config.model_id:
        snapshot = Path(marker.snapshot_path)
        if snapshot.is_dir():
            log(
                f"model ready on the volume: {marker.model_id} at revision "
                f"{marker.revision} ({marker.total_bytes / 1e9:,.1f} GB)"
            )
            return snapshot, False
        log(f"marker points at {snapshot}, which is not a directory; re-preparing")
    elif marker is not None:
        log(
            f"the volume holds {marker.model_id!r} but this worker serves "
            f"{config.model_id!r}"
        )

    if not allow_cold_download:
        raise ColdModelError(
            f"no prepared model for {config.model_id!r} on the volume mounted at "
            f"{config.persistent_root}.\n"
            f"Populate it from a CPU container first — a download does not need a GPU:\n"
            f"    modal run scripts/populate_modal_volume.py\n"
            f"To download from this GPU container anyway, deploy with "
            f"MODAL_ALLOW_COLD_DOWNLOAD=1."
        )

    log("MODAL_ALLOW_COLD_DOWNLOAD is set: downloading with the GPU attached and billing")
    try:
        outcome = prepare_model(config, log=log)
    except ModelPreparationError as exc:
        raise ColdModelError(str(exc)) from exc
    return Path(outcome.state.snapshot_path), outcome.downloaded


# -- vLLM ---------------------------------------------------------------


def launch_vllm(
    config: WorkerConfig,
    snapshot: Path,
    *,
    extra_environment: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> subprocess.Popen[bytes]:
    """Start vLLM against the local snapshot, never against a hub identifier.

    The argument vector comes from `WorkerConfig.vllm_argv`, the same source of
    truth the RunPod entrypoint uses, so the two deployments cannot drift into
    serving the same model with different flags.
    """
    argv = config.vllm_argv(snapshot)
    environment = dict(os.environ)
    environment.update(config.hub_environment())
    if extra_environment:
        environment.update(extra_environment)
    # Offline once the snapshot is local: a hub lookup on the critical path is a
    # network dependency between an idle GPU and a served token.
    environment.setdefault("HF_HUB_OFFLINE", "1")
    log("launching: " + " ".join(config.redacted_argv(snapshot)))
    return subprocess.Popen(argv, env=environment)


# -- sleep mode, which is what makes a snapshot possible ----------------


class SleepModeError(RuntimeError):
    """vLLM refused to sleep or wake. The snapshot would be worthless."""


def warmup_vllm(
    config: WorkerConfig,
    *,
    rounds: int = 3,
    timeout: float = 300.0,
    log: Callable[[str], None] = print,
) -> None:
    """Serve a few real completions before snapshotting.

    Not a health check. CUDA graphs and some Torch compilation outputs are
    produced lazily, on the first inference rather than at load, and a snapshot
    taken before them captures a container that will still have to build them
    on every restore. Three small requests are what puts that work inside the
    snapshot instead of after it.
    """
    payload = {
        "model": config.public_model_name,
        "messages": [{"role": "user", "content": "Who are you?"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    for index in range(rounds):
        response = httpx.post(
            f"{config.base_url}/v1/chat/completions", json=payload, timeout=timeout
        )
        response.raise_for_status()
        log(f"warmup {index + 1}/{rounds} ok")


def sleep_vllm(config: WorkerConfig, *, level: int = 1, timeout: float = 300.0) -> None:
    """Offload the weights to CPU memory and drop the KV cache.

    Level 1 keeps the weights in host RAM, which is what makes the snapshot a
    fixed, restorable thing rather than a photograph of 29 GB of device memory
    the next container may not be able to reproduce.

    Requires `VLLM_SERVER_DEV_MODE=1`; without it the route does not exist and
    vLLM answers 404.
    """
    try:
        response = httpx.post(f"{config.base_url}/sleep?level={level}", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SleepModeError(
            f"vLLM would not sleep: {exc}. "
            "Sleep mode needs VLLM_SERVER_DEV_MODE=1 in the image and "
            "--enable-sleep-mode on the command line."
        ) from exc


def wake_vllm(config: WorkerConfig, *, timeout: float = 300.0) -> None:
    """Bring the weights back to the GPU after a snapshot restore."""
    try:
        response = httpx.post(f"{config.base_url}/wake_up", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SleepModeError(f"vLLM would not wake up: {exc}") from exc


# -- startup accounting -------------------------------------------------


@dataclass(frozen=True, slots=True)
class StartupRecord:
    """One container's boot, broken into the parts that can differ.

    Section 24 of the specification asks for this split for a reason: a slow
    cold start is usually not Modal being slow to find a GPU. It is the weights
    moving, or the graphs compiling. Reporting one `cold_start_ms` hides which.
    """

    started_at: str
    container_id: str
    gpu: str
    model_id: str
    revision: str | None
    cold_download: bool
    volume_reload_ms: int = 0
    model_resolve_ms: int = 0
    vllm_launch_ms: int = 0
    readiness_ms: int = 0
    total_ms: int = 0
    ready: bool = False
    error: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def append_startup_record(layout: PersistentLayout, record: StartupRecord) -> Path:
    """Append one line to the Volume's startup log.

    A separate file per container would be cleaner, but Modal Volumes are
    "last write wins" on a shared file and hostile to many small ones. Appending
    a line is the operation least likely to lose data here, and the benchmark
    tolerates a lost line far better than it tolerates a lost container-id.
    """
    path = layout.logs / _STARTUP_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.as_json() + "\n")
    return path


def startup_records(layout: PersistentLayout) -> list[dict[str, object]]:
    """Every startup the Volume has seen, oldest first. Skips corrupt lines."""
    path = layout.logs / _STARTUP_LOG
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            records.append(entry)
    return records


def new_record(config: WorkerConfig, *, gpu: str, cold_download: bool) -> StartupRecord:
    """A record for the container that is starting right now."""
    return StartupRecord(
        started_at=_now_iso(),
        # Modal sets this in every container; the fallback keeps the record
        # usable when the same code is exercised outside Modal.
        container_id=os.environ.get("MODAL_TASK_ID", f"local-{os.getpid()}"),
        gpu=gpu,
        model_id=config.model_id,
        revision=config.model_revision,
        cold_download=cold_download,
    )


def monotonic_ms(since: float) -> int:
    """Milliseconds elapsed since a `time.monotonic()` reading."""
    return int((time.monotonic() - since) * 1000)
