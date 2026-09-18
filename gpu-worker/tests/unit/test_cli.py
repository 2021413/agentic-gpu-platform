"""Command line: exit codes, output contracts and secret hygiene.

The exit codes are a contract with ``entrypoint.sh``; each one is asserted in
both its nominal and its degraded case.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from worker import cli
from worker.config import PersistentLayout, WorkerConfig
from worker.gpu import GpuInfo, GpuSurvey
from worker.model_state import ModelPreparationError, ModelState, PreparationOutcome, write_marker
from worker.readiness import SmokeResult

BASE_URL = "http://127.0.0.1:8000"
MODEL = "acme/tiny-model"
API_KEY = "sk-live-must-never-be-logged"
HF_TOKEN = "hf_must_never_be_logged"

MODELS_OK = {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
COMPLETION_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "int add(int a, int b){return a+b;}"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 17},
}


def run(command: Callable[[Sequence[str] | None], int], argv: Sequence[str] = ()) -> int:
    """Run a command, turning the ``SystemExit`` of a config error into its code."""
    try:
        return command(list(argv))
    except SystemExit as exit_request:  # _load_config exits rather than returning
        return int(exit_request.code or 0)


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """No command may shell out to a real driver during a unit test."""
    monkeypatch.setattr(cli, "survey_gpus", lambda: GpuSurvey(error="nvidia-smi is not on PATH"))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, volume: Path) -> Callable[..., None]:
    """Inject a usable worker environment; overrides are applied on top."""

    def configure(**overrides: str) -> None:
        base = {
            "MODEL_ID": MODEL,
            "PERSISTENT_ROOT": str(volume),
            "MIN_FREE_DISK_GB": "0",
            "WORKER_ALLOW_EPHEMERAL_STORAGE": "1",
            "READINESS_POLL_SECONDS": "0.01",
            "READINESS_TIMEOUT_SECONDS": "0.03",
        }
        base.update(overrides)
        for name, value in base.items():
            monkeypatch.setenv(name, value)

    configure()
    return configure


@pytest.fixture
def router() -> Any:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


def _prepared_marker(volume: Path, *, model_id: str = MODEL) -> Path:
    """Write a marker describing a snapshot that really exists on the volume."""
    layout = PersistentLayout(volume)
    snapshot = layout.hub_cache / "snapshots" / "deadbeefcafe0123"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    write_marker(
        layout,
        ModelState(
            model_id=model_id,
            revision="deadbeefcafe0123",
            snapshot_path=str(snapshot),
            prepared_at="2026-01-01T00:00:00+00:00",
            total_bytes=2,
            files={"config.json": 2},
        ),
    )
    return snapshot


# ----------------------------------------------------------------------
# preflight
# ----------------------------------------------------------------------


def test_preflight_passes_on_a_usable_environment(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(cli.preflight_main) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "preflight passed" in out
    assert "configuration" in out
    assert "persistent storage" in out
    assert "not prepared on this volume yet" in out


def test_preflight_can_skip_storage_entirely(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env(PERSISTENT_ROOT=str(volume / "never-mounted"), WORKER_ALLOW_EPHEMERAL_STORAGE="0")

    assert run(cli.preflight_main, ["--skip-storage"]) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "persistent storage" not in out


def test_preflight_rejects_a_broken_configuration_with_code_2(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    env(PORT="eight thousand")

    assert run(cli.preflight_main) == cli.EXIT_CONFIG

    captured = capsys.readouterr()
    assert "variable: PORT" in captured.err
    assert "preflight passed" not in captured.out


def test_preflight_rejects_a_missing_volume_with_code_3(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env(PERSISTENT_ROOT=str(volume / "never-mounted"), WORKER_ALLOW_EPHEMERAL_STORAGE="0")

    assert run(cli.preflight_main) == cli.EXIT_STORAGE

    captured = capsys.readouterr()
    assert "no persistent volume is attached" in captured.err
    assert "preflight passed" not in captured.out


def test_preflight_rejects_a_volume_too_small_for_a_cold_start(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    env(MIN_FREE_DISK_GB="1000000")

    assert run(cli.preflight_main) == cli.EXIT_STORAGE

    assert "required to download the model" in capsys.readouterr().err


def test_a_volume_that_already_holds_the_model_may_be_nearly_full(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepared_marker(volume)
    env(MIN_FREE_DISK_GB="1000000")

    assert run(cli.preflight_main) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "preflight passed" in out
    assert f"{MODEL}@deadbeefcafe" in out


def test_preflight_warns_when_no_gpu_is_visible(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(cli.preflight_main) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "! no GPU is visible: vLLM will fail to start" in out


def test_preflight_warns_about_mismatched_and_undersized_gpus(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "survey_gpus",
        lambda: GpuSurvey(
            (
                GpuInfo(0, "NVIDIA A100-SXM4-80GB", 81920, (8, 0), "535.104.05"),
                GpuInfo(1, "NVIDIA L40S", 46068, (8, 9), "535.104.05"),
            )
        ),
    )
    env(TENSOR_PARALLEL_SIZE="4")

    assert run(cli.preflight_main) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "lacks hardware FP8" in out
    assert "not identical" in out
    assert "exceeds the 2 visible GPU(s)" in out
    assert "do not divide evenly" in out
    assert "total GPU memory" in out


# ----------------------------------------------------------------------
# prepare-model
# ----------------------------------------------------------------------


def test_check_only_reports_an_unprepared_volume_with_code_4(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(cli.prepare_main, ["--check-only"]) == cli.EXIT_MODEL

    assert "model is not prepared" in capsys.readouterr().out


def test_check_only_accepts_a_prepared_volume(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _prepared_marker(volume)

    assert run(cli.prepare_main, ["--check-only"]) == cli.EXIT_OK

    assert f"model is prepared: {snapshot}" in capsys.readouterr().out


def test_check_only_rejects_a_marker_for_another_model(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepared_marker(volume, model_id="acme/previous-model")

    assert run(cli.prepare_main, ["--check-only"]) == cli.EXIT_MODEL

    assert "model is not prepared" in capsys.readouterr().out


def test_prepare_reports_a_download_and_exits_zero(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = ModelState(
        model_id=MODEL,
        revision="deadbeefcafe0123",
        snapshot_path="/volume/snapshot",
        prepared_at="2026-01-01T00:00:00+00:00",
        total_bytes=31 * 1024**3,
        files={},
    )
    monkeypatch.setattr(
        cli,
        "prepare_model",
        lambda _config, **_kwargs: PreparationOutcome(
            state, downloaded=True, attempts=1, duration_seconds=612.0
        ),
    )

    assert run(cli.prepare_main) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "model downloaded: 31.0 GB in 612s at /volume/snapshot" in out


def test_prepare_reports_a_reuse(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = ModelState(
        model_id=MODEL,
        revision="deadbeefcafe0123",
        snapshot_path="/volume/snapshot",
        prepared_at="2026-01-01T00:00:00+00:00",
        total_bytes=0,
        files={},
    )
    monkeypatch.setattr(
        cli,
        "prepare_model",
        lambda _config, **_kwargs: PreparationOutcome(
            state, downloaded=False, attempts=0, duration_seconds=0.0
        ),
    )

    assert run(cli.prepare_main) == cli.EXIT_OK
    assert "model reused" in capsys.readouterr().out


def test_prepare_reports_a_storage_failure_with_code_3(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env(PERSISTENT_ROOT=str(volume / "never-mounted"), WORKER_ALLOW_EPHEMERAL_STORAGE="0")

    assert run(cli.prepare_main) == cli.EXIT_STORAGE

    assert "persistent storage error" in capsys.readouterr().err


def test_prepare_reports_a_model_failure_with_code_4(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(_config: object, **_kwargs: object) -> PreparationOutcome:
        raise ModelPreparationError("the hub refused", hint="check HF_TOKEN")

    monkeypatch.setattr(cli, "prepare_model", explode)

    assert run(cli.prepare_main) == cli.EXIT_MODEL

    err = capsys.readouterr().err
    assert "model preparation failed: the hub refused" in err
    assert "fix: check HF_TOKEN" in err


def test_prepare_rejects_a_broken_configuration_with_code_2(env: Callable[..., None]) -> None:
    env(MAX_MODEL_LEN="12")

    assert run(cli.prepare_main, ["--check-only"]) == cli.EXIT_CONFIG


# ----------------------------------------------------------------------
# serve-args
# ----------------------------------------------------------------------


def argv_of(captured: str) -> list[str]:
    """Decode the NUL-separated argument vector the entrypoint consumes.

    Newlines cannot be the separator: a shlex word may contain one, and a
    line-oriented reader would split it into two arguments nobody wrote.
    """
    assert captured.endswith("\0"), "the vector must be NUL-terminated"
    return captured.split("\0")[:-1]


def test_serve_args_emits_a_nul_vector_that_reads_back_identically(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    env(VLLM_EXTRA_ARGS='--enable-prefix-caching --chat-template "a b"')

    assert run(cli.serve_args_main) == cli.EXIT_OK

    expected = WorkerConfig.from_env().vllm_argv()
    printed = argv_of(capsys.readouterr().out)
    assert printed == expected
    assert printed[:3] == ["vllm", "serve", MODEL]
    # A quoted value stays one argv word even though it contains a space.
    assert "a b" in printed


def test_serve_args_points_vllm_at_the_verified_local_snapshot(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _prepared_marker(volume)
    env(MODEL_REVISION="deadbeefcafe0123")

    assert run(cli.serve_args_main) == cli.EXIT_OK

    printed = argv_of(capsys.readouterr().out)
    assert printed[2] == str(snapshot)
    # A local snapshot already is the revision; asking for it again would need
    # the network on every cold start.
    assert "--revision" not in printed


def test_serve_args_falls_back_to_the_repository_when_the_marker_is_stale(
    env: Callable[..., None], volume: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepared_marker(volume, model_id="acme/previous-model")
    env(MODEL_REVISION="abc123")

    assert run(cli.serve_args_main) == cli.EXIT_OK

    printed = argv_of(capsys.readouterr().out)
    assert printed[2] == MODEL
    assert printed[printed.index("--revision") + 1] == "abc123"


def test_serve_args_redacts_a_key_written_into_extra_args(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    """The worker no longer passes a key itself, but an operator still might."""
    env(VLLM_EXTRA_ARGS=f"--api-key {API_KEY}")

    assert run(cli.serve_args_main, ["--redacted"]) == cli.EXIT_OK

    captured = capsys.readouterr()
    printed = argv_of(captured.out)
    assert API_KEY not in captured.out
    assert printed[printed.index("--api-key") + 1] == "***"
    assert printed[:3] == ["vllm", "serve", MODEL]


def test_serve_args_never_emits_the_api_key(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    """The key reaches vLLM through the environment, never through argv.

    An argument would put the secret in /proc/1/cmdline for anything that can
    read the process table, and in whatever file the vector is marshalled
    through on its way to exec.
    """
    env(VLLM_API_KEY=API_KEY)

    assert run(cli.serve_args_main) == cli.EXIT_OK

    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert "--api-key" not in argv_of(captured.out)


def test_serve_args_rejects_a_broken_configuration_with_code_2(env: Callable[..., None]) -> None:
    env(VLLM_EXTRA_ARGS='--chat-template "unterminated')

    assert run(cli.serve_args_main) == cli.EXIT_CONFIG


def test_serve_args_round_trips_a_word_containing_a_newline(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    env(VLLM_EXTRA_ARGS='--chat-template "line one\nline two"')

    assert run(cli.serve_args_main) == cli.EXIT_OK

    expected = WorkerConfig.from_env().vllm_argv()
    assert argv_of(capsys.readouterr().out) == expected


# ----------------------------------------------------------------------
# ready
# ----------------------------------------------------------------------


def test_ready_returns_zero_once_the_model_is_served(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.get("/v1/models").respond(200, json=MODELS_OK)

    assert run(cli.ready_main) == cli.EXIT_OK

    assert f"ready after 0s, serving {MODEL}" in capsys.readouterr().out


def test_ready_returns_five_when_the_deadline_passes(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.get("/v1/models").respond(503, json={"detail": "loading"})

    assert run(cli.ready_main, ["--timeout", "0.02"]) == cli.EXIT_NOT_READY

    assert "not ready after" in capsys.readouterr().out


def test_ready_returns_five_when_something_else_is_served(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.get("/v1/models").respond(200, json={"data": [{"id": "meta-llama/Llama-3-8B"}]})

    assert run(cli.ready_main, ["--timeout", "0.02"]) == cli.EXIT_NOT_READY

    assert "meta-llama/Llama-3-8B" in capsys.readouterr().out


def test_ready_quiet_prints_only_the_verdict(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.get("/v1/models").respond(503, json={"detail": "loading"})

    assert run(cli.ready_main, ["--timeout", "0.02", "--quiet"]) == cli.EXIT_NOT_READY

    out = capsys.readouterr().out
    assert "not ready yet" not in out
    assert out.strip().startswith("not ready after")


def test_ready_rejects_a_broken_configuration_with_code_2(env: Callable[..., None]) -> None:
    env(GPU_MEMORY_UTILIZATION="2")

    assert run(cli.ready_main) == cli.EXIT_CONFIG


# ----------------------------------------------------------------------
# health
# ----------------------------------------------------------------------


def test_health_returns_zero_when_something_answers(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.get("/health").respond(200, text="")

    assert run(cli.health_main) == cli.EXIT_OK

    assert capsys.readouterr().out.strip() == "healthy"


def test_health_returns_five_when_nothing_answers(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.get("/health").mock(side_effect=httpx.ConnectError("connection refused"))

    assert run(cli.health_main) == cli.EXIT_NOT_READY

    assert "ConnectError" in capsys.readouterr().out


def test_health_rejects_a_broken_configuration_with_code_2(env: Callable[..., None]) -> None:
    env(TENSOR_PARALLEL_SIZE="0")

    assert run(cli.health_main) == cli.EXIT_CONFIG


# ----------------------------------------------------------------------
# smoke-test
# ----------------------------------------------------------------------


def test_smoke_returns_zero_on_a_real_answer(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.post("/v1/chat/completions").respond(200, json=COMPLETION_OK)

    assert run(cli.smoke_main) == cli.EXIT_OK

    captured = capsys.readouterr()
    assert "smoke test passed" in captured.out
    assert captured.err == ""


def test_smoke_returns_six_on_an_empty_answer(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads(json.dumps(COMPLETION_OK))
    payload["choices"][0]["message"]["content"] = ""
    router.post("/v1/chat/completions").respond(200, json=payload)

    assert run(cli.smoke_main) == cli.EXIT_SMOKE

    captured = capsys.readouterr()
    assert "smoke test FAILED" in captured.err
    assert "produced no output" in captured.err
    assert captured.out == ""


def test_smoke_returns_six_when_nothing_answers(
    env: Callable[..., None], router: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    router.post("/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))

    assert run(cli.smoke_main) == cli.EXIT_SMOKE

    assert "request failed" in capsys.readouterr().err


def test_smoke_forwards_its_budget(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(_config: object, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return SmokeResult(True, model=MODEL, output="ok")

    monkeypatch.setattr(cli, "smoke_test", capture)

    assert run(cli.smoke_main, ["--max-tokens", "8", "--timeout", "30"]) == cli.EXIT_OK
    assert seen == {"max_tokens": 8, "timeout": 30.0}


def test_smoke_rejects_a_broken_configuration_with_code_2(env: Callable[..., None]) -> None:
    env(MODEL_DOWNLOAD_MAX_ATTEMPTS="lots")

    assert run(cli.smoke_main) == cli.EXIT_CONFIG


def test_smoke_forwards_a_zero_token_budget(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(_config: object, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return SmokeResult(True, model=MODEL, output="ok")

    monkeypatch.setattr(cli, "smoke_test", capture)
    run(cli.smoke_main, ["--max-tokens", "0"])

    assert seen.get("max_tokens") == 0


# ----------------------------------------------------------------------
# no command leaks a secret
# ----------------------------------------------------------------------


def _assert_no_secret(captured: pytest.CaptureResult[str]) -> None:
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert HF_TOKEN not in captured.out
    assert HF_TOKEN not in captured.err


def test_no_command_leaks_a_secret_on_stdout_or_stderr(
    env: Callable[..., None],
    volume: Path,
    router: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env(VLLM_API_KEY=API_KEY, HF_TOKEN=HF_TOKEN)
    _prepared_marker(volume)
    router.get("/v1/models").respond(200, json=MODELS_OK)
    router.get("/health").respond(200, text="")
    router.post("/v1/chat/completions").respond(200, json=COMPLETION_OK)
    monkeypatch.setattr(
        cli,
        "prepare_model",
        lambda _config, **_kwargs: PreparationOutcome(
            ModelState(MODEL, "deadbeefcafe0123", "/s", "2026-01-01T00:00:00+00:00", 0, {}),
            downloaded=False,
            attempts=0,
            duration_seconds=0.0,
        ),
    )

    for command, argv in (
        (cli.preflight_main, []),
        (cli.prepare_main, []),
        (cli.prepare_main, ["--check-only"]),
        (cli.serve_args_main, ["--redacted"]),
        (cli.ready_main, []),
        (cli.health_main, []),
        (cli.smoke_main, []),
    ):
        run(command, argv)
        _assert_no_secret(capsys.readouterr())


def test_a_failing_command_does_not_leak_a_secret_either(
    env: Callable[..., None],
    volume: Path,
    router: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env(
        VLLM_API_KEY=API_KEY,
        HF_TOKEN=HF_TOKEN,
        PERSISTENT_ROOT=str(volume / "never-mounted"),
        WORKER_ALLOW_EPHEMERAL_STORAGE="0",
        PORT="70000",
    )
    router.get("/v1/models").mock(side_effect=httpx.ConnectError("refused"))

    for command in (cli.preflight_main, cli.prepare_main, cli.serve_args_main, cli.health_main):
        run(command, [])
        _assert_no_secret(capsys.readouterr())


def test_preflight_reports_the_base_image_tag_when_the_build_stamped_one(
    env: Callable[..., None], capsys: pytest.CaptureFixture[str]
) -> None:
    env(VLLM_IMAGE_TAG="vllm/vllm-openai:v0.11.0")

    assert run(cli.preflight_main, ["--skip-storage"]) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "base image" in out
    assert "vllm/vllm-openai:v0.11.0" in out
    assert "vllm " in out  # the tracked package list is printed too


def test_a_matched_pair_of_fp8_gpus_raises_no_warning(
    env: Callable[..., None], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "survey_gpus",
        lambda: GpuSurvey(
            (
                GpuInfo(0, "NVIDIA H100 80GB HBM3", 81559, (9, 0), "550.54.15"),
                GpuInfo(1, "NVIDIA H100 80GB HBM3", 81559, (9, 0), "550.54.15"),
            )
        ),
    )
    env(TENSOR_PARALLEL_SIZE="2")

    assert run(cli.preflight_main) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "  ! " not in out
    assert "preflight passed" in out
