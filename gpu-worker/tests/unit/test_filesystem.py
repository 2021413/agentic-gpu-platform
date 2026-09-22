"""Persistent volume validation, disk accounting and quota recognition."""

from __future__ import annotations

import errno
import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.conftest import requires_unprivileged
from worker.config import WorkerConfig
from worker.filesystem import (
    GIB,
    StorageError,
    directory_size_bytes,
    disk_report,
    ensure_layout,
    inspect_storage,
    is_quota_error,
    is_separate_mount,
)

# ----------------------------------------------------------------------
# ensure_layout: every refusal names the path and tells the operator what to do
# ----------------------------------------------------------------------


def test_missing_volume_is_fatal_and_explains_how_to_attach_one(tmp_path: Path) -> None:
    config = WorkerConfig(persistent_root=tmp_path / "never-mounted")

    with pytest.raises(StorageError) as caught:
        ensure_layout(config, require_mount=True)

    error = caught.value
    assert error.path == tmp_path / "never-mounted"
    assert "no persistent volume is attached" in str(error)
    assert error.hint is not None
    assert "PERSISTENT_ROOT" in error.hint
    assert "fix:" in error.render()


def test_a_file_where_the_volume_should_be_is_fatal(tmp_path: Path) -> None:
    impostor = tmp_path / "volume"
    impostor.write_text("not a directory", encoding="utf-8")
    config = WorkerConfig(persistent_root=impostor)

    with pytest.raises(StorageError) as caught:
        ensure_layout(config, require_mount=False)

    assert "is not a directory" in str(caught.value)
    assert caught.value.path == impostor


def test_a_file_where_the_volume_should_be_offers_a_hint(tmp_path: Path) -> None:
    impostor = tmp_path / "volume"
    impostor.write_text("not a directory", encoding="utf-8")
    config = WorkerConfig(persistent_root=impostor)

    with pytest.raises(StorageError) as caught:
        ensure_layout(config, require_mount=False)

    assert caught.value.hint


def test_a_root_on_the_container_filesystem_is_refused(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("worker.filesystem.is_separate_mount", lambda _path: False)
    config = WorkerConfig(persistent_root=volume)

    with pytest.raises(StorageError) as caught:
        ensure_layout(config, require_mount=True)

    assert "not a mounted volume" in str(caught.value)
    assert caught.value.hint is not None
    assert "WORKER_ALLOW_EPHEMERAL_STORAGE" in caught.value.hint


@requires_unprivileged
def test_a_read_only_volume_is_refused_with_a_useful_hint(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("worker.filesystem.is_separate_mount", lambda _path: True)
    volume.chmod(0o500)
    config = WorkerConfig(persistent_root=volume)
    try:
        with pytest.raises(StorageError) as caught:
            ensure_layout(config, require_mount=True)
    finally:
        volume.chmod(0o700)

    error = caught.value
    assert "is not writable" in str(error)
    assert "EACCES" in str(error)
    assert error.hint is not None
    assert "read-only" in error.hint


def test_require_mount_false_creates_the_root_and_the_whole_tree(tmp_path: Path) -> None:
    root = tmp_path / "ephemeral" / "volume"
    config = WorkerConfig(persistent_root=root)

    layout = ensure_layout(config, require_mount=False)

    assert root.is_dir()
    assert layout.root == root
    for directory in layout.all_directories():
        assert directory.is_dir(), directory
    assert layout.hub_cache.is_dir()


def test_ensure_layout_is_idempotent(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("worker.filesystem.is_separate_mount", lambda _path: True)
    config = WorkerConfig(persistent_root=volume)

    ensure_layout(config, require_mount=True)
    (config.layout.state / "keep-me").write_text("x", encoding="utf-8")
    ensure_layout(config, require_mount=True)

    assert (config.layout.state / "keep-me").exists()


def test_the_write_probe_leaves_nothing_behind(volume: Path) -> None:
    config = WorkerConfig(persistent_root=volume)

    ensure_layout(config, require_mount=False)

    assert list(volume.glob(".worker-write-probe")) == []


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def test_disk_report_answers_for_a_directory_that_does_not_exist_yet(tmp_path: Path) -> None:
    report = disk_report(tmp_path / "not" / "created" / "yet")

    assert report.total_bytes > 0
    assert report.path == tmp_path / "not" / "created" / "yet"
    assert 0.0 <= report.used_ratio <= 1.0
    assert "free of" in report.render()


def test_the_container_root_is_not_a_separate_mount() -> None:
    assert is_separate_mount(Path("/")) is False


def test_is_separate_mount_never_raises_on_a_hostile_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> os.stat_result:
        raise OSError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr("worker.filesystem.os.stat", refuse)

    assert is_separate_mount(Path("/")) is False


def test_inspect_storage_reports_the_cache_and_the_requirement(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("worker.filesystem.is_separate_mount", lambda _path: True)
    config = WorkerConfig(persistent_root=volume, min_free_disk_gb=1.0)

    report = inspect_storage(config, require_mount=True)

    (config.layout.hub_cache / "blob").write_bytes(b"\x00" * 4096)
    assert report.writable is True
    assert report.is_separate_mount is True
    assert report.model_cache_bytes == 0
    assert report.required_free_bytes == GIB
    assert report.has_enough_free is (report.disk.free_bytes >= GIB)
    assert any("persistent root" in line for line in report.render())


def test_inspect_storage_measures_an_existing_cache(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("worker.filesystem.is_separate_mount", lambda _path: True)
    config = WorkerConfig(persistent_root=volume)
    ensure_layout(config, require_mount=True)
    (config.layout.hub_cache / "blob").write_bytes(b"\x00" * 4096)

    report = inspect_storage(config, require_mount=True)

    assert report.model_cache_bytes == 4096
    assert report.model_cache_gb == 4096 / GIB


# ----------------------------------------------------------------------
# directory_size_bytes: one inode, counted once
# ----------------------------------------------------------------------

BLOB_SIZE = 100_000


def _hugging_face_style_cache(root: Path) -> Path:
    """A blob store with a symlinked snapshot and a hard-linked duplicate."""
    blobs = root / "blobs"
    snapshot = root / "snapshots" / "deadbeef"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    blob = blobs / "0123456789abcdef"
    blob.write_bytes(b"\x00" * BLOB_SIZE)

    # The hub links snapshots at the blobs; following them would count the model
    # twice and make a healthy volume look full.
    (snapshot / "model.safetensors").symlink_to(blob)
    os.link(blob, snapshot / "model.hardlink")
    return blob


def test_a_hard_link_is_not_counted_twice(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.write_bytes(b"\x00" * BLOB_SIZE)
    os.link(blob, tmp_path / "same-inode")

    assert directory_size_bytes(tmp_path) == BLOB_SIZE


def test_a_symlinked_snapshot_is_not_counted_twice(tmp_path: Path) -> None:
    blob = _hugging_face_style_cache(tmp_path)
    symlink = tmp_path / "snapshots" / "deadbeef" / "model.safetensors"

    total = directory_size_bytes(tmp_path)

    # Only the blob's bytes, plus the handful of bytes the symlink's own inode
    # occupies. Nothing is doubled, and nothing is tripled by the hard link.
    assert total == BLOB_SIZE + symlink.lstat().st_size
    assert total < 2 * BLOB_SIZE
    assert blob.stat().st_size == BLOB_SIZE


def test_a_symlink_loop_does_not_hang_the_walk(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "file").write_bytes(b"x" * 10)
    (tmp_path / "real" / "loop").symlink_to(tmp_path / "real", target_is_directory=True)

    assert directory_size_bytes(tmp_path) >= 10


def test_a_missing_directory_measures_zero(tmp_path: Path) -> None:
    assert directory_size_bytes(tmp_path / "absent") == 0


def test_a_dangling_symlink_is_tolerated(tmp_path: Path) -> None:
    (tmp_path / "broken").symlink_to(tmp_path / "gone")

    assert directory_size_bytes(tmp_path) == (tmp_path / "broken").lstat().st_size


def test_nested_files_are_summed(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "one").write_bytes(b"x" * 10)
    (tmp_path / "a" / "b" / "two").write_bytes(b"x" * 20)

    assert directory_size_bytes(tmp_path) == 30


# ----------------------------------------------------------------------
# is_quota_error: the one failure that must never be retried
# ----------------------------------------------------------------------


def _raised_from(cause: BaseException) -> Exception:
    """A wrapper raised with ``from``, the way a hub client reports failure."""
    try:
        raise RuntimeError("consistency check in xet download failed") from cause
    except RuntimeError as exc:
        return exc


def _raised_during(cause: BaseException) -> Exception:
    """A wrapper raised inside an ``except`` block: implicit ``__context__``."""
    try:
        raise cause
    except BaseException:
        try:
            raise RuntimeError("download failed")
        except RuntimeError as exc:
            return exc


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: OSError(errno.ENOSPC, "No space left on device"), id="ENOSPC"),
        pytest.param(lambda: OSError(errno.EDQUOT, "Disk quota exceeded"), id="EDQUOT"),
        pytest.param(lambda: RuntimeError("Disk quota exceeded"), id="message-quota"),
        pytest.param(lambda: RuntimeError("No space left on device"), id="message-enospc"),
        pytest.param(lambda: ValueError("DISK QUOTA EXCEEDED"), id="message-uppercase"),
        pytest.param(
            lambda: _raised_from(OSError(errno.ENOSPC, "No space left on device")),
            id="wrapped-cause",
        ),
        pytest.param(
            lambda: _raised_during(OSError(errno.EDQUOT, "Disk quota exceeded")),
            id="wrapped-context",
        ),
    ],
)
def test_a_full_volume_is_recognised(factory: Callable[[], BaseException]) -> None:
    assert is_quota_error(factory()) is True


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: httpx.ConnectError("[Errno 111] Connection refused"), id="connect"),
        pytest.param(lambda: httpx.ReadTimeout("timed out"), id="timeout"),
        pytest.param(
            lambda: OSError(errno.ECONNRESET, "Connection reset by peer"), id="ECONNRESET"
        ),
        pytest.param(lambda: RuntimeError("401 Client Error: Unauthorized"), id="gated-repo"),
        pytest.param(lambda: _raised_from(httpx.ConnectError("connection refused")), id="wrapped"),
        pytest.param(lambda: FileNotFoundError(errno.ENOENT, "No such file"), id="ENOENT"),
    ],
)
def test_an_ordinary_failure_is_not_a_quota_error(factory: Callable[[], BaseException]) -> None:
    assert is_quota_error(factory()) is False


def test_used_gb_is_reported_alongside_free_and_total(tmp_path: Path) -> None:
    report = disk_report(tmp_path)

    assert report.used_gb == report.used_bytes / GIB
    assert report.total_gb == report.total_bytes / GIB
    assert report.free_gb == report.free_bytes / GIB


def test_is_separate_mount_walks_up_to_an_existing_ancestor() -> None:
    assert is_separate_mount(Path("/nonexistent/deep/path")) is False


def test_a_subdirectory_blocked_by_a_file_is_reported(volume: Path) -> None:
    """A stray file where ``models/`` belongs must not read as a working volume."""
    (volume / "models").write_text("in the way", encoding="utf-8")
    config = WorkerConfig(persistent_root=volume)

    with pytest.raises(StorageError) as caught:
        ensure_layout(config, require_mount=False)

    assert "could not create" in str(caught.value)
    assert caught.value.path == volume / "models"


@requires_unprivileged
def test_a_root_that_cannot_be_created_is_reported(tmp_path: Path) -> None:
    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o500)
    config = WorkerConfig(persistent_root=parent / "volume")
    try:
        with pytest.raises(StorageError) as caught:
            ensure_layout(config, require_mount=False)
    finally:
        parent.chmod(0o700)

    assert "could not create" in str(caught.value)
    assert caught.value.path == parent / "volume"


def test_a_file_that_vanishes_mid_walk_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is live: a blob may be unlinked between listing and stat."""
    monkeypatch.setattr(
        "worker.filesystem.os.walk", lambda *_a, **_k: [(str(tmp_path), [], ["vanished"])]
    )

    assert directory_size_bytes(tmp_path) == 0


def test_a_storage_error_without_a_path_renders_just_the_message() -> None:
    assert (
        StorageError("the volume is gone").render()
        == "persistent storage error: the volume is gone"
    )
