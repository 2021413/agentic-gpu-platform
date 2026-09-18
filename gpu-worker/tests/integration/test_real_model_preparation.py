"""Model preparation against the real Hugging Face client.

The unit tests inject a fake downloader, which proves the orchestration but not
that the machinery works against the actual hub library. This does, using a
model of a few hundred kilobytes: the lock, the resume path, the marker and the
verification are identical whether the payload is 300 KB or 31 GB.

Marked ``integration`` because it needs outbound network; skipped cleanly
otherwise.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from worker.config import WorkerConfig
from worker.filesystem import ensure_layout
from worker.model_state import prepare_model, read_marker, verify_snapshot

pytestmark = pytest.mark.integration

# A few hundred kilobytes, public, stable, and maintained by the hub project
# precisely so tests do not have to move real weights around.
TINY_MODEL = os.environ.get("TEST_TINY_MODEL", "hf-internal-testing/tiny-random-gpt2")


def _network_available() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        return False
    return True


@pytest.fixture(autouse=True)
def _require_network() -> None:
    if not _network_available():
        pytest.skip("huggingface.co is unreachable")


@pytest.fixture
def config(tmp_path: Path) -> WorkerConfig:
    config = WorkerConfig(
        model_id=TINY_MODEL,
        persistent_root=tmp_path / "volume",
        min_free_disk_gb=0.1,
        download_max_attempts=2,
    )
    ensure_layout(config, require_mount=False)
    return config


def test_a_cold_start_downloads_verifies_and_marks(config: WorkerConfig) -> None:
    outcome = prepare_model(config, log=lambda _m: None)

    assert outcome.downloaded is True
    assert outcome.state.total_bytes > 0
    assert Path(outcome.state.snapshot_path).is_dir()

    marker = read_marker(config.layout)
    assert marker is not None
    assert marker.model_id == TINY_MODEL
    assert marker.files, "the marker recorded no files, so a warm boot cannot verify"

    ok, problems = verify_snapshot(Path(marker.snapshot_path), marker.files)
    assert ok, problems


def test_a_warm_restart_reuses_the_volume_without_downloading(
    config: WorkerConfig,
) -> None:
    """The property the whole persistent volume exists for."""
    prepare_model(config, log=lambda _m: None)

    def refuse(**_kwargs: object) -> str:
        raise AssertionError("a warm restart must not touch the network")

    outcome = prepare_model(config, downloader=refuse, log=lambda _m: None)

    assert outcome.reused is True
    assert outcome.downloaded is False


def test_a_truncated_shard_is_repaired_against_the_hub(config: WorkerConfig) -> None:
    """A killed download leaves short files; they must come back, not be blessed.

    The assertion deliberately compares against the size captured *before* the
    corruption. An earlier version of this test verified the repaired snapshot
    against sizes recomputed from that same snapshot, so it passed while the
    truncated length was quietly recorded as correct — the test was tautological
    and hid a defect that would have served corrupt weights to vLLM.
    """
    first = prepare_model(config, log=lambda _m: None)
    snapshot = Path(first.state.snapshot_path)

    victim = max(
        (path for path in snapshot.rglob("*") if path.is_file()),
        key=lambda path: path.stat().st_size,
    )
    blob = victim.resolve()
    original_size = blob.stat().st_size
    os.chmod(blob, 0o644)
    with blob.open("r+b") as handle:
        handle.truncate(original_size // 2)
    assert (snapshot / victim.name).stat().st_size != original_size

    prepare_model(config, log=lambda _m: None)

    assert (snapshot / victim.name).stat().st_size == original_size, (
        "the truncated file was not restored"
    )
    marker = read_marker(config.layout)
    assert marker is not None
    assert marker.verified is True
    assert marker.files[victim.name] == original_size, (
        "the marker recorded the truncated length as if it were correct"
    )


def test_a_pinned_revision_is_recorded_as_the_resolved_commit(
    config: WorkerConfig,
) -> None:
    """A marker saying 'main' would be an apparent pin on a moving branch."""
    outcome = prepare_model(config, log=lambda _m: None)

    assert len(outcome.state.revision) == 40, outcome.state.revision
    assert all(character in "0123456789abcdef" for character in outcome.state.revision)


def test_the_marker_survives_a_process_restart(config: WorkerConfig) -> None:
    """State lives on the volume, not in the process that wrote it."""
    prepare_model(config, log=lambda _m: None)

    reloaded = WorkerConfig(
        model_id=config.model_id,
        persistent_root=config.persistent_root,
        min_free_disk_gb=0.1,
    )
    marker = read_marker(reloaded.layout)
    assert marker is not None
    assert marker.matches(reloaded.model_id, None)
