"""Container-side decisions for the Modal worker, argued offline.

No Modal account, no GPU, no network. `infra/modal/runtime.py` imports no
`modal` symbol precisely so that these can run anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from infra.modal.runtime import (
    ColdModelError,
    append_startup_record,
    compile_cache_environment,
    new_record,
    resolve_model,
    startup_records,
)
from worker.config import PersistentLayout, WorkerConfig
from worker.model_state import ModelState, write_marker


def _mark_ready(layout: PersistentLayout, config: WorkerConfig, snapshot: Path) -> ModelState:
    state = ModelState(
        model_id=config.model_id,
        revision="deadbeefcafe0123",
        snapshot_path=str(snapshot),
        prepared_at="2026-09-22T00:00:00+00:00",
        total_bytes=31_200_000_000,
        files={"config.json": 24},
        verified=True,
    )
    write_marker(layout, state)
    return state


# -- the refusal that pays for itself --------------------------------------
def test_an_unprepared_volume_refuses_to_start_the_gpu(
    config: WorkerConfig, layout: PersistentLayout
) -> None:
    """Downloading 31.2 GB with an H100 attached costs about two and a half
    dollars to move bytes a CPU container moves for a fraction of a cent."""
    with pytest.raises(ColdModelError) as caught:
        resolve_model(config, allow_cold_download=False, log=lambda _m: None)

    message = str(caught.value)
    assert "modal run scripts/populate_modal_volume.py" in message, (
        "the refusal must name the command that fixes it"
    )
    assert "MODAL_ALLOW_COLD_DOWNLOAD" in message


def test_a_prepared_volume_is_used_without_touching_the_hub(
    config: WorkerConfig, layout: PersistentLayout, fake_snapshot: Path
) -> None:
    _mark_ready(layout, config, fake_snapshot)

    snapshot, downloaded = resolve_model(
        config, allow_cold_download=False, log=lambda _m: None
    )

    assert snapshot == fake_snapshot
    assert downloaded is False


def test_a_volume_holding_a_different_model_is_refused(
    make_config: Callable[..., WorkerConfig], layout: PersistentLayout, fake_snapshot: Path
) -> None:
    """Serving whatever happens to be cached is how a fleet quietly changes
    what it answers with."""
    prepared = make_config(model_id="some/other-model")
    _mark_ready(layout, prepared, fake_snapshot)
    wanted = make_config(model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")

    with pytest.raises(ColdModelError):
        resolve_model(wanted, allow_cold_download=False, log=lambda _m: None)


def test_a_marker_pointing_nowhere_is_not_trusted(
    config: WorkerConfig, layout: PersistentLayout, tmp_path: Path
) -> None:
    _mark_ready(layout, config, tmp_path / "vanished")

    with pytest.raises(ColdModelError):
        resolve_model(config, allow_cold_download=False, log=lambda _m: None)


# -- compiled artifacts ----------------------------------------------------
def test_compiled_artifacts_are_keyed_by_gpu_and_runtime(layout: PersistentLayout) -> None:
    """CUDA graphs built on an H100 are not reusable on an H200, and Modal may
    hand us either for the same `gpu="H100"` request."""
    hopper = compile_cache_environment(layout, gpu="nvidia-h100-80gb-hbm3", image_tag="vllm:1")
    h200 = compile_cache_environment(layout, gpu="nvidia-h200", image_tag="vllm:1")
    newer = compile_cache_environment(layout, gpu="nvidia-h100-80gb-hbm3", image_tag="vllm:2")

    assert hopper["VLLM_CACHE_ROOT"] != h200["VLLM_CACHE_ROOT"]
    assert hopper["VLLM_CACHE_ROOT"] != newer["VLLM_CACHE_ROOT"]
    assert hopper["TRITON_CACHE_DIR"].startswith(hopper["VLLM_CACHE_ROOT"])


def test_compiled_artifacts_live_on_the_volume(layout: PersistentLayout) -> None:
    environment = compile_cache_environment(layout, gpu="nvidia-h100-80gb-hbm3", image_tag="v")

    assert environment["VLLM_CACHE_ROOT"].startswith(str(layout.root))


# -- startup accounting ----------------------------------------------------
def test_startup_records_survive_a_round_trip(
    config: WorkerConfig, layout: PersistentLayout
) -> None:
    first = new_record(config, gpu="nvidia-h100-80gb-hbm3", cold_download=False)
    append_startup_record(layout, first)
    append_startup_record(layout, new_record(config, gpu="nvidia-h200", cold_download=True))

    records = startup_records(layout)

    assert [entry["gpu"] for entry in records] == ["nvidia-h100-80gb-hbm3", "nvidia-h200"]
    assert records[1]["cold_download"] is True


def test_a_truncated_line_does_not_lose_the_rest(
    config: WorkerConfig, layout: PersistentLayout
) -> None:
    """Modal Volumes are last-write-wins; a half-written line is survivable and
    must not take the whole benchmark with it."""
    append_startup_record(layout, new_record(config, gpu="h100", cold_download=False))
    path = layout.logs / "startup.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"gpu": "h2\n')
    append_startup_record(layout, new_record(config, gpu="h200", cold_download=False))

    assert [entry["gpu"] for entry in startup_records(layout)] == ["h100", "h200"]


def test_records_are_json_one_line_each(config: WorkerConfig, layout: PersistentLayout) -> None:
    append_startup_record(layout, new_record(config, gpu="h100", cold_download=False))
    line = (layout.logs / "startup.jsonl").read_text(encoding="utf-8").strip()

    assert json.loads(line)["model_id"] == config.model_id


def test_no_records_yet_is_not_an_error(layout: PersistentLayout) -> None:
    assert startup_records(layout) == []
