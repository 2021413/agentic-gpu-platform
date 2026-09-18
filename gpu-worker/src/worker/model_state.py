"""Model acquisition: lock, resume, verify, mark.

This is the part that decides whether a restart costs three seconds or thirty
gigabytes. Four rules shape it:

* **One downloader at a time.** Two processes racing on the same cache produce
  corrupt blobs and double the bill. An advisory file lock on the volume makes
  the second one wait.
* **Resume, never restart.** An interrupted download leaves partial blobs the
  hub client can continue from; deleting them to "start clean" is how a flaky
  network turns into an unbounded bill.
* **Verify before trusting.** A snapshot directory existing proves nothing. The
  marker records every file and its size, so verification works offline.
* **Never delete a good model automatically.** A corrupt cache is reported, and
  removing it is an explicit operator action.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from worker.config import PersistentLayout, WorkerConfig
from worker.filesystem import (
    GIB,
    StorageError,
    directory_size_bytes,
    disk_report,
    is_quota_error,
)

__all__ = [
    "LockTimeoutError",
    "ModelPreparationError",
    "ModelState",
    "PreparationOutcome",
    "download_lock",
    "prepare_model",
    "read_marker",
    "verify_snapshot",
    "write_marker",
]

MARKER_VERSION = 2

# Weights and configuration only. Fetching every file would also pull the
# duplicate .pt/.gguf variants some repositories carry.
ALLOW_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
)


class ModelPreparationError(RuntimeError):
    """The model could not be made available. Carries an operator diagnostic."""

    def __init__(self, message: str, *, hint: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.hint = hint
        self.retryable = retryable

    def render(self) -> str:
        lines = [f"model preparation failed: {self}"]
        if self.hint:
            lines.append(f"  fix: {self.hint}")
        return "\n".join(lines)


class LockTimeoutError(ModelPreparationError):
    """Another process held the download lock for longer than allowed."""


@dataclass(frozen=True, slots=True)
class ModelState:
    """The contents of ``state/model-ready.json``.

    Written only after a snapshot has been downloaded *and* verified, so its
    presence is the single source of truth for "this volume already has it".
    """

    model_id: str
    revision: str
    snapshot_path: str
    prepared_at: str
    total_bytes: int
    files: Mapping[str, int] = field(default_factory=dict)
    verified: bool = True
    marker_version: int = MARKER_VERSION
    vllm_image: str | None = None

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GIB

    def matches(self, model_id: str, revision: str | None) -> bool:
        """Whether this marker describes what the worker was asked to serve.

        An unpinned request matches any resident revision: re-downloading 31 GB
        because ``main`` moved is rarely what an operator wants, and pinning
        ``MODEL_REVISION`` is the documented way to demand an exact one.
        """
        if self.model_id != model_id:
            return False
        return revision is None or self.revision == revision

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ModelState:
        return cls(
            model_id=str(payload["model_id"]),
            revision=str(payload["revision"]),
            snapshot_path=str(payload["snapshot_path"]),
            prepared_at=str(payload["prepared_at"]),
            total_bytes=int(payload.get("total_bytes", 0)),
            files={str(k): int(v) for k, v in (payload.get("files") or {}).items()},
            verified=bool(payload.get("verified", True)),
            marker_version=int(payload.get("marker_version", 0)),
            vllm_image=payload.get("vllm_image"),
        )


# ----------------------------------------------------------------------
# marker
# ----------------------------------------------------------------------
def read_marker(layout: PersistentLayout) -> ModelState | None:
    """Load the readiness marker, or ``None`` if absent or unreadable.

    A corrupt marker is treated as absent rather than fatal: the snapshot is
    re-verified, which is cheap compared with refusing to boot.
    """
    path = layout.model_ready_marker
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        state = ModelState.from_mapping(payload)
    except (KeyError, TypeError, ValueError):
        return None
    return state if state.marker_version == MARKER_VERSION else None


def write_marker(layout: PersistentLayout, state: ModelState) -> Path:
    """Write the marker atomically.

    Written to a sibling and renamed: a marker truncated by a Pod dying
    mid-write would be worse than no marker at all, because it claims a model is
    ready when only half of it is.
    """
    target = layout.model_ready_marker
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(state.to_json(), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


# ----------------------------------------------------------------------
# lock
# ----------------------------------------------------------------------
@contextmanager
def download_lock(
    layout: PersistentLayout,
    *,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
    on_wait: Callable[[float, str], None] | None = None,
) -> Iterator[None]:
    """Hold the exclusive download lock, or fail after ``timeout_seconds``.

    ``flock`` is advisory but released by the kernel when the holder dies, which
    is exactly right here: a Pod killed mid-download must not leave a lock file
    that blocks every future boot. The holder's identity is written inside for
    diagnostics only — never trusted for correctness.
    """
    path = layout.download_lock
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    handle = path.open("a+")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockTimeoutError(
                        f"another process has held the model download lock for more than "
                        f"{timeout_seconds:g}s ({_lock_holder(handle)})",
                        hint=(
                            "a previous download may still be running on this volume; "
                            "raise MODEL_LOCK_TIMEOUT_SECONDS or stop the other Pod"
                        ),
                        retryable=True,
                    ) from None
                if on_wait is not None:
                    on_wait(remaining, _lock_holder(handle))
                time.sleep(min(poll_seconds, remaining))
        handle.seek(0)
        handle.truncate()
        handle.write(f"{socket.gethostname()} pid={os.getpid()} at={_now()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _lock_holder(handle: Any) -> str:
    try:
        handle.seek(0)
        content = handle.read().strip()
    except OSError:
        return "holder unknown"
    return content or "holder unknown"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------
def verify_snapshot(snapshot: Path, files: Mapping[str, int]) -> tuple[bool, list[str]]:
    """Check every recorded file is present at its recorded size.

    Sizes rather than hashes: hashing 31 GB on every boot would add minutes to a
    restart, while a truncated or missing shard — the failure that actually
    happens after a killed download — changes the size.
    """
    problems: list[str] = []
    if not snapshot.is_dir():
        return False, [f"snapshot directory {snapshot} is missing"]
    for name, expected in files.items():
        candidate = snapshot / name
        try:
            actual = candidate.stat().st_size
        except OSError:
            problems.append(f"missing: {name}")
            continue
        if actual != expected:
            problems.append(f"truncated: {name} is {actual} bytes, expected {expected}")
    return not problems, problems


def snapshot_files(snapshot: Path) -> dict[str, int]:
    """Every regular file under a snapshot, by path relative to it."""
    result: dict[str, int] = {}
    for current, _dirs, names in os.walk(snapshot, followlinks=True):
        for name in names:
            candidate = Path(current) / name
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            result[str(candidate.relative_to(snapshot))] = size
    return result


def _wanted(name: str) -> bool:
    """Whether a repository file is one this worker actually downloads."""
    return any(fnmatch(name, pattern) for pattern in ALLOW_PATTERNS)


def hub_file_sizes(config: WorkerConfig, *, log: Callable[[str], None]) -> dict[str, int] | None:
    """The sizes the hub reports for this revision, or ``None`` if unreachable.

    This is the only *external* source of truth available. Without it, checking
    a snapshot against sizes measured from that same snapshot proves nothing —
    a truncated shard simply gets recorded as the correct length.
    """
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - reads HF_HOME at import

        info = HfApi().model_info(
            config.model_id,
            revision=config.model_revision,
            files_metadata=True,
            token=config.hf_token.reveal() or None,
        )
    except Exception as exc:
        log(f"could not read file metadata from the hub ({type(exc).__name__}: {exc})")
        return None

    sizes: dict[str, int] = {}
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None)
        if size and _wanted(sibling.rfilename):
            sizes[sibling.rfilename] = int(size)
    return sizes or None


def authoritative_sizes(
    config: WorkerConfig, *, previous: ModelState | None, log: Callable[[str], None]
) -> dict[str, int] | None:
    """Expected file sizes, from outside the snapshot being verified.

    Preference order: the hub, then the sizes recorded by an earlier preparation
    that was itself verified. A marker written without an authoritative source
    is not trusted as one.
    """
    sizes = hub_file_sizes(config, log=log)
    if sizes:
        return sizes
    if previous is not None and previous.verified and previous.files:
        log(
            "hub unreachable; verifying against the sizes recorded by the last verified preparation"
        )
        return dict(previous.files)
    return None


def _mismatches(snapshot: Path, expected: Mapping[str, int]) -> list[str]:
    """Files that are absent or the wrong length, by name."""
    broken: list[str] = []
    for name, size in expected.items():
        candidate = snapshot / name
        try:
            actual = candidate.stat().st_size
        except OSError:
            broken.append(name)
            continue
        if actual != size:
            broken.append(name)
    return broken


def _discard(snapshot: Path, names: Sequence[str], *, log: Callable[[str], None]) -> None:
    """Remove exactly the corrupt files so the hub client fetches them again.

    Narrow and loud by design. The rule elsewhere is that a cached model is
    never deleted automatically; this is the one exception, and it applies only
    to files proven to differ from what the hub says they are — leaving them in
    place would mean serving corrupt weights forever, since the hub client skips
    any file that already exists.
    """
    for name in names:
        link = snapshot / name
        blob = link.resolve() if link.is_symlink() else link
        for path in {link, blob}:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log(f"could not remove {path}: {exc}")
        log(f"discarded corrupt file so it can be fetched again: {name}")


# ----------------------------------------------------------------------
# preparation
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PreparationOutcome:
    """What preparation did, so the caller can report it truthfully."""

    state: ModelState
    downloaded: bool
    attempts: int
    duration_seconds: float

    @property
    def reused(self) -> bool:
        return not self.downloaded


Downloader = Callable[..., str]


SizesProvider = Callable[[WorkerConfig, "ModelState | None"], Mapping[str, int] | None]


def prepare_model(
    config: WorkerConfig,
    *,
    downloader: Downloader | None = None,
    sizes: SizesProvider | None = None,
    log: Callable[[str], None] = print,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PreparationOutcome:
    """Make the model available on the volume, and prove it.

    Returns without downloading when the marker matches and the snapshot still
    verifies — the warm-restart path the whole design exists to protect.
    """
    layout = config.layout
    started = now()

    previous = read_marker(layout)
    existing = _already_prepared(config, log=log)
    if existing is not None:
        return PreparationOutcome(existing, downloaded=False, attempts=0, duration_seconds=0.0)

    with download_lock(
        layout,
        timeout_seconds=config.lock_timeout_seconds,
        on_wait=lambda remaining, holder: log(
            f"waiting for the model download lock held by {holder} "
            f"({remaining:.0f}s before giving up)"
        ),
    ):
        # Re-check under the lock: while we waited, the other process may have
        # finished exactly the download we were about to start.
        existing = _already_prepared(config, log=log)
        if existing is not None:
            return PreparationOutcome(
                existing, downloaded=False, attempts=0, duration_seconds=now() - started
            )

        _require_free_space(config, log=log)
        # Injectable so a unit test can state the truth instead of reaching the
        # hub; the default is the only real source of truth there is.
        provider: SizesProvider = sizes or (
            lambda cfg, prev: authoritative_sizes(cfg, previous=prev, log=log)
        )
        expected = provider(config, previous)
        snapshot, attempts = _download_with_retries(
            config, downloader=downloader, log=log, now=now, sleep=sleep
        )
        snapshot, attempts, verified = _repair_until_correct(
            config,
            snapshot=snapshot,
            expected=expected,
            attempts=attempts,
            downloader=downloader,
            log=log,
            now=now,
            sleep=sleep,
        )
        files = snapshot_files(snapshot)
        total = sum(files.values())
        state = ModelState(
            model_id=config.model_id,
            revision=_resolved_revision(snapshot),
            snapshot_path=str(snapshot),
            prepared_at=_now(),
            total_bytes=total,
            files=files,
            verified=verified,
            vllm_image=os.environ.get("VLLM_IMAGE_TAG"),
        )
        write_marker(layout, state)
        log(f"model ready: {total / GIB:,.1f} GB at {snapshot}")
        return PreparationOutcome(
            state,
            downloaded=True,
            attempts=attempts,
            duration_seconds=now() - started,
        )


def _repair_until_correct(
    config: WorkerConfig,
    *,
    snapshot: Path,
    expected: Mapping[str, int] | None,
    attempts: int,
    downloader: Downloader | None,
    log: Callable[[str], None],
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[Path, int, bool]:
    """Check the snapshot against an external truth, and repair what differs.

    The hub client skips any file that already exists, so a blob truncated by a
    killed download is never re-fetched on its own. Without this step the
    truncated length would simply be recorded as correct, the marker would claim
    the model was verified, and every later boot would agree — while vLLM loaded
    corrupt weights.
    """
    if expected is None:
        log(
            "WARNING: no authoritative file sizes are available, so this snapshot "
            "cannot be proven intact; it is recorded as unverified and will be "
            "checked again on the next boot"
        )
        return snapshot, attempts, False

    broken = _mismatches(snapshot, expected)
    if not broken:
        log(f"verified {len(expected)} file(s) against the hub")
        return snapshot, attempts, True

    log(f"{len(broken)} file(s) do not match what the hub reports; repairing")
    _discard(snapshot, broken, log=log)
    snapshot, retry_attempts = _download_with_retries(
        config, downloader=downloader, log=log, now=now, sleep=sleep
    )
    attempts += retry_attempts

    broken = _mismatches(snapshot, expected)
    if broken:
        raise ModelPreparationError(
            f"{len(broken)} file(s) still do not match the hub after re-downloading: "
            + ", ".join(broken[:5]),
            hint=(
                "the volume may be failing or full. Nothing was marked ready, so the "
                "next boot will try again rather than serve corrupt weights."
            ),
        )
    log("repair succeeded; the snapshot now matches the hub")
    return snapshot, attempts, True


def _already_prepared(config: WorkerConfig, *, log: Callable[[str], None]) -> ModelState | None:
    """The warm path: a marker that matches and a snapshot that still verifies."""
    marker = read_marker(config.layout)
    if marker is None:
        return None
    if not marker.matches(config.model_id, config.model_revision):
        log(
            f"marker describes {marker.model_id}@{marker.revision[:12]}, "
            f"which is not what was requested; preparing again"
        )
        return None
    if not marker.verified:
        log("the recorded preparation was never checked against the hub; preparing again")
        return None
    ok, problems = verify_snapshot(Path(marker.snapshot_path), marker.files)
    if ok:
        log(
            f"reusing the model already on this volume: {marker.total_gb:,.1f} GB "
            f"at {marker.snapshot_path} (prepared {marker.prepared_at})"
        )
        return marker
    log("the cached model failed verification and will be completed:")
    for problem in problems[:10]:
        log(f"  {problem}")
    return None


def _require_free_space(config: WorkerConfig, *, log: Callable[[str], None]) -> None:
    report = disk_report(config.layout.hub_cache)
    log(f"disk before download: {report.render()}")
    required = config.min_free_disk_gb * GIB
    if report.free_bytes < required:
        raise ModelPreparationError(
            f"only {report.free_gb:,.1f} GB free on {report.path}, "
            f"{config.min_free_disk_gb:g} GB required",
            hint=(
                "grow the RunPod network volume, remove an old model revision from "
                f"{config.layout.hub_cache}, or lower MIN_FREE_DISK_GB if you know "
                "the model is smaller"
            ),
        )


def _download_with_retries(
    config: WorkerConfig,
    *,
    downloader: Downloader | None,
    log: Callable[[str], None],
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[Path, int]:
    """Download, resuming on transient failure and stopping on a full disk.

    Returns the snapshot and how many attempts it really took. Reporting a
    constant 1 would hide a retry storm from the operator watching the boot.
    """
    fetch = downloader if downloader is not None else _snapshot_download
    last: Exception | None = None

    for attempt in range(1, config.download_max_attempts + 1):
        started = now()
        log(
            f"downloading {config.model_id}"
            f"@{config.model_revision or 'main'} (attempt {attempt}/"
            f"{config.download_max_attempts})"
        )
        try:
            path = fetch(
                repo_id=config.model_id,
                revision=config.model_revision,
                cache_dir=str(config.layout.hub_cache),
                token=config.hf_token.reveal() or None,
            )
        except Exception as exc:
            if is_quota_error(exc):
                report = disk_report(config.layout.hub_cache)
                raise ModelPreparationError(
                    f"the volume ran out of space while downloading: {exc}",
                    hint=(
                        f"{report.render()}. The model needs about 31 GB plus room to "
                        "resume; grow the volume rather than retrying, which would "
                        "only fill the same disk more slowly"
                    ),
                ) from exc
            last = exc
            if attempt >= config.download_max_attempts:
                break
            delay = config.download_backoff_seconds * (2 ** (attempt - 1))
            log(f"download failed ({type(exc).__name__}: {exc}); resuming in {delay:.0f}s")
            sleep(delay)
            continue
        log(f"download finished in {now() - started:.0f}s")
        return Path(path), attempt

    raise ModelPreparationError(
        f"download failed after {config.download_max_attempts} attempts: {last}",
        hint=(
            "check outbound network access and, for a gated repository, that HF_TOKEN "
            "is set and authorised"
        ),
        retryable=True,
    ) from last


def _snapshot_download(**kwargs: Any) -> str:
    """The real hub client.

    Imported inside the function on purpose, and this is not a style slip:
    ``huggingface_hub`` reads ``HF_HOME`` and ``HF_HUB_CACHE`` at *import* time.
    Importing it before the persistent layout has been exported would pin the
    cache to the container filesystem, which is the exact failure this whole
    package exists to prevent. It also keeps the unit tests free of the hub
    stack.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - see docstring

    return str(
        snapshot_download(
            repo_id=kwargs["repo_id"],
            revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
            # Weights and config only. Fetching every file would also pull the
            # duplicate .pt/.gguf variants some repositories carry.
            allow_patterns=list(ALLOW_PATTERNS),
            max_workers=8,
        )
    )


def _resolved_revision(snapshot: Path) -> str:
    """The commit a snapshot directory stands for.

    The hub lays out ``snapshots/<commit-sha>/``, so the directory name is the
    resolved revision even when the request was for a moving ref.
    """
    return snapshot.name


def clear_model_cache(config: WorkerConfig) -> int:
    """Remove the cached model. Only ever called explicitly by an operator.

    Returns the number of bytes freed. Deliberately not wired into any automatic
    recovery path: silently deleting 31 GB because a check failed is how a flaky
    disk turns into an hour of downtime and a surprise bill.
    """
    cache = config.layout.hub_cache
    freed = directory_size_bytes(cache)
    shutil.rmtree(cache, ignore_errors=True)
    config.layout.model_ready_marker.unlink(missing_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return freed


def storage_error_hint(exc: StorageError) -> str:
    return exc.render()
