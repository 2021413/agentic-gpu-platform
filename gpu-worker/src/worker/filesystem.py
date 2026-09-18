"""Persistent storage validation and disk accounting.

The single most expensive failure this image can have is downloading 31 GB into
a container filesystem that disappears with the Pod. Everything here exists to
make that impossible rather than unlikely: the volume is proven mounted and
writable *before* anything large is attempted, and a quota failure is reported
as a quota failure instead of being retried forever.
"""

from __future__ import annotations

import errno
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from worker.config import PersistentLayout, WorkerConfig

__all__ = [
    "DiskReport",
    "StorageError",
    "StorageReport",
    "directory_size_bytes",
    "disk_report",
    "ensure_layout",
    "inspect_storage",
    "is_quota_error",
]

GIB = 1024**3


class StorageError(RuntimeError):
    """The persistent volume cannot be used. Always fatal, never retried."""

    def __init__(self, message: str, *, path: Path | None = None, hint: str | None = None):
        super().__init__(message)
        self.path = path
        self.hint = hint

    def render(self) -> str:
        lines = [f"persistent storage error: {self}"]
        if self.path is not None:
            lines.append(f"  path: {self.path}")
        if self.hint:
            lines.append(f"  fix:  {self.hint}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DiskReport:
    """Capacity of the filesystem backing a path."""

    path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GIB

    @property
    def used_gb(self) -> float:
        return self.used_bytes / GIB

    @property
    def free_gb(self) -> float:
        return self.free_bytes / GIB

    @property
    def used_ratio(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes else 0.0

    def render(self) -> str:
        return (
            f"{self.free_gb:,.1f} GB free of {self.total_gb:,.1f} GB "
            f"({self.used_ratio * 100:.0f}% used) on {self.path}"
        )


@dataclass(frozen=True, slots=True)
class StorageReport:
    """What preflight found on the volume."""

    layout: PersistentLayout
    disk: DiskReport
    writable: bool
    is_separate_mount: bool
    model_cache_bytes: int
    required_free_bytes: int

    @property
    def model_cache_gb(self) -> float:
        return self.model_cache_bytes / GIB

    @property
    def has_enough_free(self) -> bool:
        return self.disk.free_bytes >= self.required_free_bytes

    def render(self) -> list[str]:
        return [
            f"persistent root        {self.layout.root}",
            f"separate mount         {'yes' if self.is_separate_mount else 'NO (container fs!)'}",
            f"writable               {'yes' if self.writable else 'NO'}",
            f"capacity               {self.disk.render()}",
            f"model cache in use     {self.model_cache_gb:,.1f} GB",
            f"required free          {self.required_free_bytes / GIB:,.1f} GB",
        ]


def disk_report(path: Path) -> DiskReport:
    """Capacity of the filesystem that holds ``path``.

    Walks up to the nearest existing ancestor: asking about a directory that
    preflight has not created yet is normal, and should still answer.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return DiskReport(
        path=path, total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free
    )


def is_separate_mount(path: Path) -> bool:
    """Whether ``path`` is a mount point distinct from the container root.

    A volume that failed to attach usually looks like an ordinary empty
    directory, which is exactly how 31 GB ends up in a layer that evaporates.
    Comparing device numbers is what tells the two apart.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return os.stat(probe).st_dev != os.stat("/").st_dev
    except OSError:
        return False


def _probe_writable(directory: Path) -> tuple[bool, str | None]:
    """Write, flush and delete a marker. Nothing else proves writability.

    A read-only bind mount and a quota-exhausted volume both pass ``os.access``;
    only an actual write tells them apart.
    """
    probe = directory / ".worker-write-probe"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with probe.open("wb") as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, str(exc.errno)) if exc.errno else "OSError"
        return False, f"{name}: {exc.strerror or exc}"
    finally:
        probe.unlink(missing_ok=True)
    return True, None


def ensure_layout(config: WorkerConfig, *, require_mount: bool = True) -> PersistentLayout:
    """Validate the volume and create the directory tree.

    Fails fast and loudly. Every branch below is a real deployment mistake that
    would otherwise surface half an hour later as a confusing download error.
    """
    layout = config.layout
    root = layout.root

    if not root.exists():
        if require_mount:
            raise StorageError(
                f"{root} does not exist, so no persistent volume is attached",
                path=root,
                hint=(
                    "attach a RunPod network volume with mount path "
                    f"{root}, or set PERSISTENT_ROOT to where it is mounted"
                ),
            )
        # Ephemeral mode is an explicit opt-in for throwaway runs; there the root
        # is ours to create, because no volume is expected to provide it.
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                f"could not create {root}: {exc.strerror or exc}", path=root
            ) from exc
    if not root.is_dir():
        raise StorageError(
            f"{root} exists but is not a directory",
            path=root,
            hint=(
                "something else already occupies the mount point — usually a file "
                "created by an earlier run before the volume was attached. Remove it "
                "and restart the Pod with the volume mounted there."
            ),
        )

    if require_mount and not is_separate_mount(root):
        raise StorageError(
            f"{root} is on the container filesystem, not a mounted volume",
            path=root,
            hint=(
                "model weights would be lost on every Pod restart and would fill "
                "the container disk; attach the volume, or set "
                "WORKER_ALLOW_EPHEMERAL_STORAGE=1 to accept that for a throwaway test"
            ),
        )

    writable, detail = _probe_writable(root)
    if not writable:
        raise StorageError(
            f"{root} is not writable ({detail})",
            path=root,
            hint="check the volume is not read-only and that the quota is not exhausted",
        )

    for directory in layout.all_directories():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                f"could not create {directory}: {exc.strerror or exc}", path=directory
            ) from exc
    return layout


def directory_size_bytes(path: Path) -> int:
    """Bytes actually consumed, counting each inode once.

    The Hugging Face cache is a blob store with symlinked snapshots; following
    links would report the model twice and make a healthy volume look full.
    """
    if not path.exists():
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    for current, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            candidate = Path(current) / name
            try:
                info = candidate.lstat()
            except OSError:
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += info.st_size
    return total


def inspect_storage(config: WorkerConfig, *, require_mount: bool = True) -> StorageReport:
    """Everything preflight needs to decide whether to proceed.

    Creates the tree as part of inspecting it. Reporting on a layout that does
    not exist yet would answer questions about the wrong filesystem — and
    "writable: no" for a directory nobody has created is a misleading way to
    describe a perfectly healthy volume.
    """
    layout = ensure_layout(config, require_mount=require_mount)
    root = layout.root
    writable = _probe_writable(root)[0]
    return StorageReport(
        layout=layout,
        disk=disk_report(root),
        writable=writable,
        is_separate_mount=is_separate_mount(root),
        model_cache_bytes=directory_size_bytes(layout.hub_cache),
        required_free_bytes=int(config.min_free_disk_gb * GIB),
    )


_QUOTA_ERRNOS = frozenset({errno.EDQUOT, errno.ENOSPC})
_QUOTA_MARKERS = ("disk quota exceeded", "no space left on device", "quota exceeded")


def is_quota_error(exc: BaseException) -> bool:
    """Whether a failure means "the volume is full".

    Recognised explicitly because it is the one download failure that must never
    be retried: the second attempt fills the same disk, slower.
    """
    if isinstance(exc, OSError) and exc.errno in _QUOTA_ERRNOS:
        return True
    text = str(exc).lower()
    if any(marker in text for marker in _QUOTA_MARKERS):
        return True
    cause = exc.__cause__ or exc.__context__
    return bool(cause is not None and cause is not exc and is_quota_error(cause))
