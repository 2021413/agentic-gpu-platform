"""Model acquisition: marker, verification, retries, quota and the file lock."""

from __future__ import annotations

import errno
import fcntl
import json
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import FakeClock, Recorder

from worker.config import ConfigError, PersistentLayout, Secret, WorkerConfig
from worker.filesystem import StorageError
from worker.model_state import (
    MARKER_VERSION,
    LockTimeoutError,
    ModelPreparationError,
    ModelState,
    clear_model_cache,
    download_lock,
    prepare_model,
    read_marker,
    snapshot_files,
    storage_error_hint,
    verify_snapshot,
    write_marker,
)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _state(snapshot: Path, **overrides: Any) -> ModelState:
    payload: dict[str, Any] = {
        "model_id": "acme/tiny-model",
        "revision": "deadbeefcafe0123",
        "snapshot_path": str(snapshot),
        "prepared_at": "2026-01-01T00:00:00+00:00",
        "total_bytes": sum(snapshot_files(snapshot).values()),
        "files": snapshot_files(snapshot),
    }
    payload.update(overrides)
    return ModelState(**payload)


def no_hub(_config: WorkerConfig, _previous: ModelState | None) -> None:
    """No external source of truth, because no unit test may touch the network.

    With an injected downloader there is no real repository to ask, so the
    snapshot cannot be proven intact and the marker records that honestly
    (``verified=False``). Tests that need an authoritative table state it
    themselves.
    """
    return None


class SpyDownloader:
    """A downloader that records its calls and follows a scripted outcome."""

    def __init__(self, *, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)


# ----------------------------------------------------------------------
# marker
# ----------------------------------------------------------------------


def test_marker_round_trips_through_the_filesystem(
    layout: PersistentLayout, fake_snapshot: Path
) -> None:
    state = _state(fake_snapshot)

    target = write_marker(layout, state)

    assert target == layout.model_ready_marker
    reloaded = read_marker(layout)
    assert reloaded == state
    assert reloaded is not None
    assert reloaded.marker_version == MARKER_VERSION
    assert reloaded.total_gb == state.total_bytes / 1024**3


def test_marker_is_written_atomically(layout: PersistentLayout, fake_snapshot: Path) -> None:
    """The visible file is renamed into place; no temporary is left behind."""
    write_marker(layout, _state(fake_snapshot))

    leftovers = list(layout.state.glob("*.tmp"))
    assert leftovers == []
    payload = json.loads(layout.model_ready_marker.read_text(encoding="utf-8"))
    assert payload["model_id"] == "acme/tiny-model"
    assert payload["verified"] is True


def test_marker_overwrite_never_exposes_a_half_written_file(
    layout: PersistentLayout, fake_snapshot: Path
) -> None:
    write_marker(layout, _state(fake_snapshot))
    write_marker(layout, _state(fake_snapshot, model_id="acme/other"))

    reloaded = read_marker(layout)
    assert reloaded is not None
    assert reloaded.model_id == "acme/other"
    assert list(layout.state.glob("*.tmp")) == []


def test_write_marker_creates_the_state_directory(
    config: WorkerConfig, fake_snapshot: Path
) -> None:
    layout = config.layout
    assert not layout.state.exists()

    write_marker(layout, _state(fake_snapshot))

    assert layout.model_ready_marker.is_file()


def test_an_absent_marker_reads_as_none(layout: PersistentLayout) -> None:
    assert read_marker(layout) is None


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("truncated-json", '{"model_id": "acme/tiny-model", "revi'),
        ("not-json", "this volume was formatted by a cosmic ray"),
        ("json-but-not-an-object", "[1, 2, 3]"),
        ("missing-keys", '{"model_id": "acme/tiny-model"}'),
        (
            "wrong-types",
            '{"model_id": "a", "revision": "b", "snapshot_path": "c",'
            ' "prepared_at": "d", "total_bytes": "enormous"}',
        ),
        ("empty", ""),
    ],
)
def test_a_corrupt_marker_is_treated_as_absent(
    layout: PersistentLayout, label: str, content: str
) -> None:
    layout.model_ready_marker.write_text(content, encoding="utf-8")

    assert read_marker(layout) is None, label


def test_invalid_utf8_in_the_marker_is_treated_as_absent(layout: PersistentLayout) -> None:
    layout.model_ready_marker.write_bytes(b"\xff\xfe\x00 not utf-8")

    assert read_marker(layout) is None


@pytest.mark.parametrize("version", [0, 1, 99])
def test_a_marker_of_an_unknown_version_is_treated_as_absent(
    layout: PersistentLayout, fake_snapshot: Path, version: int
) -> None:
    payload = json.loads(_state(fake_snapshot).to_json())
    payload["marker_version"] = version
    layout.model_ready_marker.write_text(json.dumps(payload), encoding="utf-8")

    assert read_marker(layout) is None


def test_a_marker_directory_reads_as_absent_rather_than_raising(
    layout: PersistentLayout,
) -> None:
    layout.model_ready_marker.mkdir(parents=True)

    assert read_marker(layout) is None


# ----------------------------------------------------------------------
# matches
# ----------------------------------------------------------------------


def test_an_unpinned_request_matches_any_resident_revision(fake_snapshot: Path) -> None:
    state = _state(fake_snapshot, revision="0123456789ab")

    assert state.matches("acme/tiny-model", None) is True


def test_a_pinned_request_demands_the_exact_revision(fake_snapshot: Path) -> None:
    state = _state(fake_snapshot, revision="0123456789ab")

    assert state.matches("acme/tiny-model", "0123456789ab") is True
    assert state.matches("acme/tiny-model", "ffffffffffff") is False


def test_a_different_model_never_matches(fake_snapshot: Path) -> None:
    state = _state(fake_snapshot)

    assert state.matches("acme/other-model", None) is False
    assert state.matches("acme/other-model", "deadbeefcafe0123") is False


# ----------------------------------------------------------------------
# verify_snapshot
# ----------------------------------------------------------------------


def test_an_intact_snapshot_verifies(fake_snapshot: Path) -> None:
    ok, problems = verify_snapshot(fake_snapshot, snapshot_files(fake_snapshot))

    assert ok is True
    assert problems == []


def test_a_missing_file_is_detected(fake_snapshot: Path) -> None:
    files = snapshot_files(fake_snapshot)
    (fake_snapshot / "config.json").unlink()

    ok, problems = verify_snapshot(fake_snapshot, files)

    assert ok is False
    assert problems == ["missing: config.json"]


def test_a_truncated_file_is_detected(fake_snapshot: Path) -> None:
    files = snapshot_files(fake_snapshot)
    shard = fake_snapshot / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"\x00" * 10)

    ok, problems = verify_snapshot(fake_snapshot, files)

    assert ok is False
    assert problems == [
        "truncated: model-00001-of-00001.safetensors is 10 bytes, expected 2048",
    ]


def test_a_missing_snapshot_directory_is_detected(tmp_path: Path) -> None:
    ok, problems = verify_snapshot(tmp_path / "gone", {"a": 1})

    assert ok is False
    assert "is missing" in problems[0]


def test_snapshot_files_are_relative_and_recursive(fake_snapshot: Path) -> None:
    (fake_snapshot / "nested").mkdir()
    (fake_snapshot / "nested" / "tokenizer.json").write_bytes(b"{}")

    files = snapshot_files(fake_snapshot)

    assert files["nested/tokenizer.json"] == 2
    assert files["config.json"] > 0
    assert all(not Path(name).is_absolute() for name in files)


# ----------------------------------------------------------------------
# prepare_model: the warm path
# ----------------------------------------------------------------------


def _explode(**_kwargs: Any) -> str:
    raise AssertionError("the downloader must not be called on the warm path")


def test_a_valid_marker_and_an_intact_snapshot_skip_the_download(
    config: WorkerConfig, layout: PersistentLayout, fake_snapshot: Path, recorder: Recorder
) -> None:
    write_marker(layout, _state(fake_snapshot))
    spy = SpyDownloader(outcomes=[fake_snapshot])

    outcome = prepare_model(config, sizes=no_hub, downloader=spy, log=recorder)

    assert outcome.downloaded is False
    assert outcome.reused is True
    assert outcome.attempts == 0
    assert spy.calls == []
    assert outcome.state.snapshot_path == str(fake_snapshot)
    assert "reusing the model already on this volume" in recorder.text


def test_the_warm_path_does_not_even_take_the_lock(
    config: WorkerConfig, layout: PersistentLayout, fake_snapshot: Path, recorder: Recorder
) -> None:
    write_marker(layout, _state(fake_snapshot))
    handle = layout.download_lock.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        outcome = prepare_model(config, sizes=no_hub, downloader=_explode, log=recorder)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    assert outcome.reused is True


def test_a_marker_for_another_model_forces_a_new_preparation(
    make_config: Callable[..., WorkerConfig],
    layout: PersistentLayout,
    fake_snapshot: Path,
    recorder: Recorder,
    clock: FakeClock,
) -> None:
    write_marker(layout, _state(fake_snapshot, model_id="acme/previous-model"))
    config = make_config(model_id="acme/tiny-model")
    spy = SpyDownloader(outcomes=[fake_snapshot])

    outcome = prepare_model(
        config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
    )

    assert outcome.downloaded is True
    assert len(spy.calls) == 1
    assert "which is not what was requested" in recorder.text


def test_a_failing_verification_forces_a_new_preparation(
    config: WorkerConfig,
    layout: PersistentLayout,
    fake_snapshot: Path,
    recorder: Recorder,
    clock: FakeClock,
) -> None:
    write_marker(layout, _state(fake_snapshot))
    (fake_snapshot / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 3)
    spy = SpyDownloader(outcomes=[fake_snapshot])

    outcome = prepare_model(
        config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
    )

    assert outcome.downloaded is True
    assert "failed verification" in recorder.text
    assert any("truncated" in line for line in recorder.lines)


# ----------------------------------------------------------------------
# prepare_model: the cold path
# ----------------------------------------------------------------------


def test_a_cold_volume_downloads_then_writes_the_marker(
    config: WorkerConfig, fake_snapshot: Path, recorder: Recorder, clock: FakeClock
) -> None:
    spy = SpyDownloader(outcomes=[fake_snapshot])

    outcome = prepare_model(
        config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
    )

    assert outcome.downloaded is True
    assert clock.sleeps == []
    assert spy.calls == [
        {
            "repo_id": "acme/tiny-model",
            "revision": None,
            "cache_dir": str(config.layout.hub_cache),
            "token": None,
        }
    ]
    marker = read_marker(config.layout)
    assert marker is not None
    assert marker.model_id == "acme/tiny-model"
    # Nothing authoritative was available, so the snapshot is recorded as
    # unproven and will be checked again rather than trusted.
    assert marker.verified is False
    assert marker.snapshot_path == str(fake_snapshot)
    assert marker.files == snapshot_files(fake_snapshot)
    assert marker.total_bytes == sum(snapshot_files(fake_snapshot).values())
    # An unpinned request records the commit the snapshot directory stands for.
    assert marker.revision == fake_snapshot.name


def test_the_pinned_revision_and_the_token_reach_the_downloader(
    make_config: Callable[..., WorkerConfig],
    fake_snapshot: Path,
    recorder: Recorder,
    clock: FakeClock,
) -> None:
    config = make_config(model_revision="abc123", hf_token=Secret("hf_secret"))
    spy = SpyDownloader(outcomes=[fake_snapshot])

    prepare_model(
        config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
    )

    assert spy.calls[0]["revision"] == "abc123"
    assert spy.calls[0]["token"] == "hf_secret"
    marker = read_marker(config.layout)
    assert marker is not None
    # The resolved snapshot directory, not the requested string: recording
    # "main" would look like a pin while tracking a moving branch.
    assert marker.revision == fake_snapshot.name


def test_a_volume_without_room_refuses_before_downloading(
    make_config: Callable[..., WorkerConfig], recorder: Recorder, clock: FakeClock
) -> None:
    config = make_config(min_free_disk_gb=10**6)
    spy = SpyDownloader(outcomes=["unused"])

    with pytest.raises(ModelPreparationError) as caught:
        prepare_model(
            config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
        )

    assert spy.calls == []
    assert "required" in str(caught.value)
    assert caught.value.hint is not None
    assert "MIN_FREE_DISK_GB" in caught.value.hint


# ----------------------------------------------------------------------
# prepare_model: retries, backoff, quota
# ----------------------------------------------------------------------


def test_two_transient_failures_are_resumed_with_exponential_backoff(
    config: WorkerConfig, fake_snapshot: Path, recorder: Recorder, clock: FakeClock
) -> None:
    spy = SpyDownloader(
        outcomes=[
            ConnectionResetError("connection reset by peer"),
            TimeoutError("read timed out"),
            fake_snapshot,
        ]
    )

    outcome = prepare_model(
        config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
    )

    assert outcome.downloaded is True
    assert len(spy.calls) == 3
    # 5s then 10s: doubling, and observed through the injected sleep rather than
    # by waiting for it.
    assert clock.sleeps == [5.0, 10.0]
    assert "attempt 3/3" in recorder.text
    assert read_marker(config.layout) is not None


def test_the_outcome_reports_how_many_attempts_were_really_needed(
    config: WorkerConfig, fake_snapshot: Path, recorder: Recorder, clock: FakeClock
) -> None:
    spy = SpyDownloader(
        outcomes=[
            ConnectionResetError("connection reset by peer"),
            TimeoutError("read timed out"),
            fake_snapshot,
        ]
    )

    outcome = prepare_model(
        config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
    )

    assert len(spy.calls) == 3
    assert outcome.attempts == 3


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(OSError(errno.ENOSPC, "No space left on device"), id="ENOSPC"),
        pytest.param(OSError(errno.EDQUOT, "Disk quota exceeded"), id="EDQUOT"),
        pytest.param(RuntimeError("Disk quota exceeded"), id="message"),
    ],
)
def test_a_full_volume_fails_immediately_without_a_single_retry(
    config: WorkerConfig,
    recorder: Recorder,
    clock: FakeClock,
    failure: BaseException,
) -> None:
    spy = SpyDownloader(outcomes=[failure])

    with pytest.raises(ModelPreparationError) as caught:
        prepare_model(
            config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
        )

    assert len(spy.calls) == 1, "a full disk must never be retried"
    assert clock.sleeps == []
    assert "ran out of space" in str(caught.value)
    assert caught.value.retryable is False
    assert caught.value.hint is not None
    assert "grow the volume" in caught.value.hint
    assert read_marker(config.layout) is None


def test_exhausting_the_attempts_reports_a_retryable_failure(
    config: WorkerConfig, recorder: Recorder, clock: FakeClock
) -> None:
    spy = SpyDownloader(outcomes=[ConnectionResetError("connection reset by peer")])

    with pytest.raises(ModelPreparationError) as caught:
        prepare_model(
            config, sizes=no_hub, downloader=spy, log=recorder, now=clock.now, sleep=clock.sleep
        )

    assert len(spy.calls) == config.download_max_attempts == 3
    assert clock.sleeps == [5.0, 10.0]
    assert caught.value.retryable is True
    assert "after 3 attempts" in str(caught.value)
    assert caught.value.hint is not None
    assert "HF_TOKEN" in caught.value.hint
    assert "fix:" in caught.value.render()
    assert read_marker(config.layout) is None


def test_a_quota_failure_wrapped_by_the_hub_client_still_stops_immediately(
    config: WorkerConfig, recorder: Recorder, clock: FakeClock
) -> None:
    def fetch(**_kwargs: Any) -> str:
        raise RuntimeError("consistency check failed") from OSError(
            errno.ENOSPC, "No space left on device"
        )

    with pytest.raises(ModelPreparationError):
        prepare_model(
            config, sizes=no_hub, downloader=fetch, log=recorder, now=clock.now, sleep=clock.sleep
        )

    assert clock.sleeps == []


# ----------------------------------------------------------------------
# the lock
# ----------------------------------------------------------------------


def test_the_lock_records_its_holder_for_diagnostics(layout: PersistentLayout) -> None:
    with download_lock(layout, timeout_seconds=1.0):
        content = layout.download_lock.read_text(encoding="utf-8")

    assert f"pid={os.getpid()}" in content


def test_the_lock_is_reentrant_across_sequential_acquisitions(layout: PersistentLayout) -> None:
    with download_lock(layout, timeout_seconds=1.0):
        pass
    with download_lock(layout, timeout_seconds=1.0):
        pass


def test_a_held_lock_times_out_and_names_the_holder(layout: PersistentLayout) -> None:
    """``flock`` is per open file description, so a second handle really conflicts."""
    holder = layout.download_lock.open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    holder.write("another-pod pid=4242 at=2026-01-01T00:00:00+00:00\n")
    holder.flush()
    try:
        with (
            pytest.raises(LockTimeoutError) as caught,
            download_lock(layout, timeout_seconds=0.05, poll_seconds=0.01),
        ):
            pytest.fail("the lock must not be granted while another handle holds it")
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

    error = caught.value
    assert isinstance(error, ModelPreparationError)
    assert error.retryable is True
    assert "another-pod pid=4242" in str(error)
    assert error.hint is not None
    assert "MODEL_LOCK_TIMEOUT_SECONDS" in error.hint


def test_the_wait_callback_reports_the_remaining_time(layout: PersistentLayout) -> None:
    holder = layout.download_lock.open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    waits: list[tuple[float, str]] = []
    try:
        with (
            pytest.raises(LockTimeoutError),
            download_lock(
                layout,
                timeout_seconds=0.05,
                poll_seconds=0.01,
                on_wait=lambda remaining, who: waits.append((remaining, who)),
            ),
        ):
            pass
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

    assert waits
    assert all(0 < remaining <= 0.05 for remaining, _ in waits)
    assert waits[0][1] == "holder unknown"


def test_the_lock_is_released_when_the_body_raises(layout: PersistentLayout) -> None:
    with pytest.raises(ZeroDivisionError), download_lock(layout, timeout_seconds=1.0):
        raise ZeroDivisionError

    # A second acquisition proves the handle was unlocked and closed.
    with download_lock(layout, timeout_seconds=0.5):
        pass


def test_prepare_model_surfaces_a_lock_timeout(
    config: WorkerConfig, layout: PersistentLayout, recorder: Recorder, clock: FakeClock
) -> None:
    holder = layout.download_lock.open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(LockTimeoutError):
            prepare_model(
                config,
                sizes=no_hub,
                downloader=_explode,
                log=recorder,
                now=clock.now,
                sleep=clock.sleep,
            )
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


# -- two real processes ------------------------------------------------


def _child_expects_to_be_blocked(root: str, queue: Any) -> None:  # pragma: no cover - subprocess
    layout = PersistentLayout(Path(root))
    try:
        with download_lock(layout, timeout_seconds=0.4, poll_seconds=0.02):
            queue.put("acquired")
    except LockTimeoutError:
        queue.put("blocked")
    except Exception as exc:
        queue.put(f"error:{type(exc).__name__}:{exc}")


def _child_holds_then_dies(root: str, queue: Any) -> None:  # pragma: no cover - subprocess
    layout = PersistentLayout(Path(root))
    handle = layout.download_lock.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.write(f"child pid={os.getpid()}\n")
    handle.flush()
    queue.put("acquired")
    time.sleep(60)


def test_two_real_processes_cannot_download_at_once(layout: PersistentLayout) -> None:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()

    with download_lock(layout, timeout_seconds=1.0):
        child = context.Process(target=_child_expects_to_be_blocked, args=(str(layout.root), queue))
        child.start()
        verdict = queue.get(timeout=10)
        child.join(timeout=10)

    assert verdict == "blocked", verdict
    assert child.exitcode == 0


def test_a_lock_whose_holder_dies_is_released(layout: PersistentLayout) -> None:
    """The property that makes ``flock`` the right primitive on a shared volume."""
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=_child_holds_then_dies, args=(str(layout.root), queue))
    child.start()
    try:
        assert queue.get(timeout=10) == "acquired"
        # Confirm the lock really is held while the child lives.
        with (
            pytest.raises(LockTimeoutError),
            download_lock(layout, timeout_seconds=0.05, poll_seconds=0.01),
        ):
            pass
        os.kill(child.pid or 0, signal.SIGKILL)
        child.join(timeout=10)
    finally:
        if child.is_alive():  # pragma: no cover - only on a failed test
            child.kill()
            child.join(timeout=10)

    assert child.exitcode == -signal.SIGKILL
    # The kernel dropped the dead holder's lock: no stale file blocks the boot.
    with download_lock(layout, timeout_seconds=5.0, poll_seconds=0.02):
        pass


# ----------------------------------------------------------------------
# clear_model_cache
# ----------------------------------------------------------------------


def test_clearing_the_cache_reports_what_it_freed(
    config: WorkerConfig, layout: PersistentLayout, fake_snapshot: Path
) -> None:
    write_marker(layout, _state(fake_snapshot))
    before = sum(snapshot_files(fake_snapshot).values())

    freed = clear_model_cache(config)

    assert freed >= before
    assert layout.hub_cache.is_dir()
    assert list(layout.hub_cache.iterdir()) == []
    assert read_marker(layout) is None


def test_a_dangling_link_in_a_snapshot_is_skipped(fake_snapshot: Path) -> None:
    (fake_snapshot / "tokenizer.json").symlink_to(fake_snapshot / "blob-that-was-deleted")

    files = snapshot_files(fake_snapshot)

    assert "tokenizer.json" not in files
    assert "config.json" in files


def test_storage_error_hint_delegates_to_the_rendered_error() -> None:
    error = StorageError("the volume is gone", path=Path("/runpod-volume"), hint="attach it")

    assert storage_error_hint(error) == error.render()


def _child_prepares_then_releases(root: str, snapshot: str, queue: Any) -> None:
    # pragma: no cover - subprocess
    layout = PersistentLayout(Path(root))
    with download_lock(layout, timeout_seconds=5.0, poll_seconds=0.02):
        queue.put("locked")
        time.sleep(0.3)
        write_marker(
            layout,
            ModelState(
                model_id="acme/tiny-model",
                revision="deadbeefcafe0123",
                snapshot_path=snapshot,
                prepared_at="2026-01-01T00:00:00+00:00",
                total_bytes=sum(snapshot_files(Path(snapshot)).values()),
                files=snapshot_files(Path(snapshot)),
            ),
        )


def test_the_winner_of_the_lock_race_spares_the_loser_the_download(
    make_config: Callable[..., WorkerConfig],
    layout: PersistentLayout,
    fake_snapshot: Path,
    recorder: Recorder,
) -> None:
    """The re-check under the lock is what stops two Pods downloading 31 GB twice."""
    config = make_config(lock_timeout_seconds=15.0)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(
        target=_child_prepares_then_releases,
        args=(str(layout.root), str(fake_snapshot), queue),
    )
    child.start()
    try:
        assert queue.get(timeout=10) == "locked"
        outcome = prepare_model(config, sizes=no_hub, downloader=_explode, log=recorder)
    finally:
        child.join(timeout=10)
        if child.is_alive():  # pragma: no cover - only on a failed test
            child.kill()
            child.join(timeout=10)

    assert child.exitcode == 0
    assert outcome.downloaded is False
    assert outcome.duration_seconds > 0
    assert "waiting for the model download lock" in recorder.text


def test_a_preparation_error_without_a_hint_renders_just_the_message() -> None:
    assert ModelPreparationError("no").render() == "model preparation failed: no"


def test_a_zero_attempt_budget_never_tries_at_all(
    make_config: Callable[..., WorkerConfig], recorder: Recorder, clock: FakeClock
) -> None:
    """A budget of zero is refused at construction, not discovered at boot.

    It used to be accepted, producing a worker that never attempted a single
    download and then reported "download failed after 0 attempts: None".
    """
    del recorder, clock
    with pytest.raises(ConfigError) as caught:
        make_config(download_max_attempts=0)

    assert caught.value.variable == "MODEL_DOWNLOAD_MAX_ATTEMPTS"
