"""Deployment profiles, and the mistakes they refuse to deploy.

Every refusal below corresponds to a way of spending money by accident. None of
them needs a Modal account: `ModalWorkerConfig` is a dataclass, and the point of
keeping it one is that the expensive decisions can be argued with offline.
"""

from __future__ import annotations

import pytest

from infra.modal.config import (
    DEV,
    PROD,
    ModalConfigError,
    ModalWorkerConfig,
    active_config,
)


def test_dev_scales_to_zero_and_caps_the_fleet_at_one() -> None:
    """A loop that fans out ten requests must not provision ten H100s."""
    assert DEV.min_containers == 0
    assert DEV.max_containers == 1


def test_prod_also_starts_at_zero() -> None:
    """Nothing has yet measured a latency requirement worth a warm H100."""
    assert PROD.min_containers == 0


def test_a_startup_timeout_below_a_cold_vllm_start_is_refused() -> None:
    """Modal's own default is 30s, which kills the container mid-load.

    The loop that follows is invisible: the container dies, the autoscaler
    starts another, and the client only ever sees 503.
    """
    with pytest.raises(ModalConfigError) as caught:
        ModalWorkerConfig(profile="dev", startup_timeout=60)

    assert "startup_timeout" in caught.value.render()


def test_an_idle_window_longer_than_an_hour_is_refused() -> None:
    with pytest.raises(ModalConfigError, match="idle GPU"):
        ModalWorkerConfig(profile="dev", scaledown_window=7200)


def test_prod_refuses_an_unauthenticated_url() -> None:
    """A public Server URL is an H100 anyone who finds it can bill."""
    with pytest.raises(ModalConfigError, match="unauthenticated"):
        ModalWorkerConfig(profile="prod", unauthenticated=True)

    # The same setting is allowed in dev, where it is a debugging convenience.
    assert ModalWorkerConfig(profile="dev", unauthenticated=True).unauthenticated


def test_gpu_snapshots_require_cpu_snapshots() -> None:
    with pytest.raises(ModalConfigError, match="enable_memory_snapshot"):
        ModalWorkerConfig(profile="dev", enable_gpu_snapshot=True)


def test_max_below_min_is_refused() -> None:
    with pytest.raises(ModalConfigError, match="below min_containers"):
        ModalWorkerConfig(profile="prod", min_containers=2, max_containers=1)


def test_the_benchmark_gpu_refuses_modals_free_h200_upgrade() -> None:
    """A distribution measured across H100 and H200 is two distributions."""
    assert ModalWorkerConfig(profile="dev", gpu="H100").benchmark_gpu == "H100!"
    assert ModalWorkerConfig(profile="dev", gpu="H100!").benchmark_gpu == "H100!"


def test_unset_options_are_omitted_rather_than_passed_as_none() -> None:
    kwargs = ModalWorkerConfig(profile="dev", target_concurrency=None).as_server_kwargs()

    assert "target_concurrency" not in kwargs
    assert "compute_region" not in kwargs, "pinning a region multiplies the GPU price"
    assert kwargs["unauthenticated"] is False


def test_gpu_snapshots_travel_as_an_experimental_option() -> None:
    kwargs = ModalWorkerConfig(
        profile="dev", enable_memory_snapshot=True, enable_gpu_snapshot=True
    ).as_server_kwargs()

    assert kwargs["enable_memory_snapshot"] is True
    assert kwargs["experimental_options"] == {"enable_gpu_snapshot": True}


def test_a_profile_is_selected_by_name() -> None:
    assert active_config({"MODAL_PROFILE": "prod"}).profile == "prod"
    assert active_config({}).profile == "dev", "the safe default is the capped one"


def test_an_unknown_profile_names_the_known_ones() -> None:
    with pytest.raises(ModalConfigError) as caught:
        active_config({"MODAL_PROFILE": "staging"})

    assert "dev" in caught.value.render() and "prod" in caught.value.render()


def test_one_field_can_be_overridden_for_a_benchmark() -> None:
    """Section 37 asks for the same worker at four scaledown windows."""
    config = active_config({"MODAL_PROFILE": "dev", "MODAL_SCALEDOWN_WINDOW": "300"})

    assert config.scaledown_window == 300
    assert config.max_containers == 1, "the rest of the profile is untouched"


def test_max_containers_can_be_unset_explicitly() -> None:
    assert active_config({"MODAL_MAX_CONTAINERS": "none"}).max_containers is None


def test_a_malformed_override_names_the_variable() -> None:
    with pytest.raises(ModalConfigError) as caught:
        active_config({"MODAL_SCALEDOWN_WINDOW": "a minute"})

    assert "MODAL_SCALEDOWN_WINDOW" in caught.value.render()


def test_describe_states_the_cost_of_a_warm_container() -> None:
    lines = "\n".join(ModalWorkerConfig(profile="prod", min_containers=1).describe())

    assert "bills 1 GPU(s) continuously" in lines
