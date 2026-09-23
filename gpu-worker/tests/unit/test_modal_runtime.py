"""Container-side decisions for the Modal worker, argued offline.

No Modal account, no GPU, no network. `infra/modal/runtime.py` imports no
`modal` symbol precisely so that these can run anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from infra.modal.runtime import (
    ColdModelError,
    SleepModeError,
    append_startup_record,
    compile_cache_environment,
    new_record,
    resolve_model,
    scratch_environment,
    sleep_vllm,
    startup_records,
    wake_vllm,
    warmup_vllm,
)
from tests.conftest import Recorder
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


# -- scratch space ---------------------------------------------------------
def test_temporary_files_do_not_live_on_the_volume(layout: PersistentLayout) -> None:
    """vLLM binds ZeroMQ IPC sockets under TMPDIR, and a Modal Volume is a FUSE
    filesystem with no Unix domain sockets. The first real cold start died on
    `ZMQError: Operation not supported` a minute into loading the weights."""
    tmpdir = scratch_environment()["TMPDIR"]

    assert not tmpdir.startswith(str(layout.root)), (
        "TMPDIR on the volume kills the engine after the weights start moving"
    )


def test_the_caches_that_must_stay_on_the_volume_still_do(
    layout: PersistentLayout,
) -> None:
    """The scratch exception must not drag the expensive caches with it."""
    environment = dict(layout.environment())
    environment.update(scratch_environment())

    on_volume = {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"}
    for name in on_volume:
        assert environment[name].startswith(str(layout.root)), name


# -- warmup, sleep and wake: what makes a memory snapshot worth taking -----
#
# These three run against a simulated vLLM, the same way `test_readiness.py`
# does, because every one of their failure modes is an HTTP answer: a route
# that does not exist, a half-warmed engine, a server that went away.

BASE_URL = "http://127.0.0.1:8000"

WARMUP_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "acme/tiny-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}


@pytest.fixture
def router() -> Any:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


def test_warmup_asks_for_the_name_vllm_actually_serves(
    make_config: Callable[..., WorkerConfig], router: Any, recorder: Recorder
) -> None:
    """With `--served-model-name` set, vLLM 404s any request naming the raw
    model id. A warmup that sent the id would fail every round and abort the
    deploy — or, worse, be made to pass by ignoring its own errors."""
    config = make_config(model_id="acme/tiny-model", served_model_name="acme-public")
    route = router.post("/v1/chat/completions").respond(200, json=WARMUP_OK)

    warmup_vllm(config, rounds=1, log=recorder)

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "acme-public"
    assert body["messages"][0]["role"] == "user"


def test_warmup_serves_every_round_it_was_asked_for(
    config: WorkerConfig, router: Any, recorder: Recorder
) -> None:
    """The rounds are the whole point: CUDA graphs are built lazily on the
    first inferences, so a warmup that quietly does fewer than it was told
    leaves that work outside the snapshot and in every restore instead."""
    route = router.post("/v1/chat/completions").respond(200, json=WARMUP_OK)

    warmup_vllm(config, rounds=5, log=recorder)

    assert route.call_count == 5
    assert recorder.lines[-1] == "warmup 5/5 ok"


def test_warmup_does_three_rounds_unless_told_otherwise(
    config: WorkerConfig, router: Any, recorder: Recorder
) -> None:
    """`app.py` calls this with no `rounds`; the default is what ships."""
    route = router.post("/v1/chat/completions").respond(200, json=WARMUP_OK)

    warmup_vllm(config, log=recorder)

    assert route.call_count == 3


def test_a_rejected_warmup_round_stops_before_the_snapshot(
    config: WorkerConfig, router: Any, recorder: Recorder
) -> None:
    """Snapshotting a half-warmed engine is the expensive mistake: every later
    container restores from it, so one 500 here would be paid back on every
    cold start until someone reran the deploy."""
    router.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=WARMUP_OK),
            httpx.Response(500, text="engine core died"),
            httpx.Response(200, json=WARMUP_OK),
        ]
    )

    with pytest.raises(httpx.HTTPStatusError):
        warmup_vllm(config, rounds=3, log=recorder)

    assert recorder.lines == ["warmup 1/3 ok"], "the third round must never be sent"


def test_sleep_hits_the_sleep_route_at_level_one_by_default(
    config: WorkerConfig, router: Any
) -> None:
    """Level 1 keeps the weights in host RAM. A snapshot taken at any other
    level captures device memory the restoring container may not reproduce."""
    route = router.post("/sleep").respond(200, json={"message": "ok"})

    sleep_vllm(config)

    assert str(route.calls.last.request.url) == f"{BASE_URL}/sleep?level=1"


def test_sleep_forwards_the_level_it_was_given(config: WorkerConfig, router: Any) -> None:
    """A level silently pinned to 1 would drop the caller's level-2 request on
    the floor and snapshot far more memory than asked for."""
    route = router.post("/sleep").respond(200, json={"message": "ok"})

    sleep_vllm(config, level=2)

    assert str(route.calls.last.request.url) == f"{BASE_URL}/sleep?level=2"


def test_a_404_from_sleep_names_the_two_settings_that_bring_the_route_back(
    config: WorkerConfig, router: Any
) -> None:
    """404 is the exact shape of a forgotten VLLM_SERVER_DEV_MODE: the route
    does not exist rather than failing. Nothing in that status says why, so
    the error has to name both settings or the next person reads vLLM source."""
    router.post("/sleep").respond(404, json={"detail": "Not Found"})

    with pytest.raises(SleepModeError) as caught:
        sleep_vllm(config)

    message = str(caught.value)
    assert "VLLM_SERVER_DEV_MODE=1" in message
    assert "--enable-sleep-mode" in message
    assert "404" in message, "the status is what tells you which of the two is missing"


def test_wake_hits_the_wake_up_route(config: WorkerConfig, router: Any) -> None:
    """A restored container serves nothing until the weights are back on the
    GPU, so the route name is load-bearing: a typo is a worker that 404s once
    and then answers every request from an asleep engine."""
    route = router.post("/wake_up").respond(200, json={"message": "ok"})

    wake_vllm(config)

    assert str(route.calls.last.request.url) == f"{BASE_URL}/wake_up"


@pytest.mark.parametrize("status", [404, 500, 503])
def test_a_wake_that_fails_is_raised_not_logged(
    config: WorkerConfig, router: Any, status: int
) -> None:
    """Restoring a snapshot and failing to wake leaves a container that looks
    healthy and answers nothing. It has to fail loudly, at `@modal.enter()`,
    where Modal kills it in seconds instead of routing traffic to it."""
    router.post("/wake_up").respond(status, text="no")

    with pytest.raises(SleepModeError) as caught:
        wake_vllm(config)

    assert "would not wake up" in str(caught.value)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (sleep_vllm, "would not sleep"),
        (wake_vllm, "would not wake up"),
    ],
)
def test_a_dead_server_is_a_sleep_mode_error_not_a_raw_httpx_error(
    config: WorkerConfig,
    router: Any,
    call: Callable[[WorkerConfig], None],
    expected: str,
) -> None:
    """An engine that crashed during sleep answers with a closed socket, not a
    status, so `raise_for_status` never runs. `app.py` catches SleepModeError;
    an httpx exception escaping past it would skip the diagnosis entirely and
    surface as a bare traceback in Modal's log."""
    router.post("/sleep").mock(side_effect=httpx.ConnectError("connection refused"))
    router.post("/wake_up").mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(SleepModeError) as caught:
        call(config)

    assert expected in str(caught.value)
    assert isinstance(caught.value.__cause__, httpx.HTTPError)
