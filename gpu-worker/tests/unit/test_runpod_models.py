"""The pure half of the deployer: request bodies, environments, URL choice.

No network, no RunPod, no GPU. Everything here is a function of its arguments,
which is the point of keeping these decisions out of the HTTP client.
"""

from __future__ import annotations

import pytest

from runpod_deployer.models import (
    DEFAULT_MIN_FREE_DISK_GB,
    PROXY_TIMEOUT_SECONDS,
    ExposedPort,
    PodSpec,
    PodState,
    SecretValue,
    SpecError,
    WorkerSettings,
    build_pod_environment,
    choose_access_url,
    default_ports,
)

IMAGE = "registry.example/gpu-worker:1.0.0"


def a_spec(**changes: object) -> PodSpec:
    base = {
        "image_name": IMAGE,
        "name": "worker-1",
        "gpu_type_ids": ("NVIDIA H100 80GB HBM3",),
        "network_volume_id": "vol123",
        "ports": default_ports(8000, expose="both"),
        "worker": WorkerSettings(),
    }
    base.update(changes)
    return PodSpec(**base)  # type: ignore[arg-type]


# -- environment --------------------------------------------------------
def test_environment_separates_secrets_from_injectable_values() -> None:
    settings = WorkerSettings(
        model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        max_model_len=8192,
        hf_token=SecretValue("hf_realtoken"),
        vllm_api_key=SecretValue("sk-worker"),
    )
    env = build_pod_environment(settings)

    assert env.public["MODEL_ID"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    assert env.public["MAX_MODEL_LEN"] == "8192"
    assert "HF_TOKEN" not in env.public
    assert "VLLM_API_KEY" not in env.public
    assert env.secret_names() == ("HF_TOKEN", "VLLM_API_KEY")

    # revealed only for the request body...
    assert env.as_api_env()["HF_TOKEN"] == "hf_realtoken"
    assert env.as_api_env()["VLLM_API_KEY"] == "sk-worker"
    # ...never for anything a human or a log will see
    redacted = env.redacted()
    assert redacted["HF_TOKEN"] == "***"
    assert redacted["VLLM_API_KEY"] == "***"
    assert "hf_realtoken" not in repr(settings)
    assert "hf_realtoken" not in str(settings.hf_token)


def test_environment_omits_unset_optional_variables() -> None:
    env = build_pod_environment(WorkerSettings())
    assert "MODEL_REVISION" not in env.public
    assert "SERVED_MODEL_NAME" not in env.public
    assert "VLLM_EXTRA_ARGS" not in env.public
    assert env.secret_names() == ()
    assert env.redacted() == env.public


def test_extra_args_are_quoted_into_one_variable() -> None:
    env = build_pod_environment(
        WorkerSettings(extra_args=("--enable-prefix-caching", "--swap-space", "4 GB"))
    )
    assert env.public["VLLM_EXTRA_ARGS"] == "--enable-prefix-caching --swap-space '4 GB'"


# -- spec validation ----------------------------------------------------
def test_mount_path_must_match_the_worker_persistent_root() -> None:
    with pytest.raises(SpecError, match="container disk"):
        a_spec(volume_mount_path="/workspace")


def test_worker_port_must_be_exposed() -> None:
    with pytest.raises(SpecError, match="not exposed"):
        a_spec(ports=(ExposedPort(9999, "http"),))


def test_a_pod_without_ports_is_refused() -> None:
    with pytest.raises(SpecError, match="no exposed port"):
        a_spec(ports=())


def test_empty_image_is_refused() -> None:
    with pytest.raises(SpecError, match="imageName"):
        a_spec(image_name="  ")


# -- request body -------------------------------------------------------
def test_create_body_is_exactly_what_runpod_documents() -> None:
    spec = a_spec(
        data_center_ids=("EU-RO-1",),
        container_disk_in_gb=60,
        worker=WorkerSettings(hf_token=SecretValue("hf_secret")),
    )
    body = spec.to_create_body()

    assert body == {
        "name": "worker-1",
        "imageName": IMAGE,
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuCount": 1,
        "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
        "dataCenterIds": ["EU-RO-1"],
        "containerDiskInGb": 60,
        "volumeMountPath": "/runpod-volume",
        "networkVolumeId": "vol123",
        "ports": ["8000/http", "8000/tcp"],
        "interruptible": False,
        "env": {
            "MODEL_ID": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            "PERSISTENT_ROOT": "/runpod-volume",
            "PORT": "8000",
            "MAX_MODEL_LEN": "16384",
            "GPU_MEMORY_UTILIZATION": "0.9",
            "TENSOR_PARALLEL_SIZE": "1",
            "MIN_FREE_DISK_GB": f"{DEFAULT_MIN_FREE_DISK_GB:g}",
            "READINESS_TIMEOUT_SECONDS": "1800",
            "HF_TOKEN": "hf_secret",
        },
    }


def test_create_body_without_a_network_volume_asks_for_a_pod_volume() -> None:
    body = a_spec(network_volume_id=None, volume_in_gb=100).to_create_body()
    assert "networkVolumeId" not in body
    assert body["volumeInGb"] == 100


def test_community_cloud_with_tcp_asks_for_a_public_ip() -> None:
    body = a_spec(cloud_type="COMMUNITY").to_create_body()
    assert body["supportPublicIp"] is True
    assert a_spec(cloud_type="SECURE").to_create_body().get("supportPublicIp") is None


def test_describe_never_prints_a_secret() -> None:
    spec = a_spec(worker=WorkerSettings(hf_token=SecretValue("hf_topsecret")))
    text = "\n".join(spec.describe())
    assert "hf_topsecret" not in text
    assert "HF_TOKEN" in text


# -- pod state ----------------------------------------------------------
PAYLOAD = {
    "id": "abc123",
    "name": "worker-1",
    "desiredStatus": "RUNNING",
    "image": IMAGE,
    "publicIp": "213.173.99.7",
    "portMappings": {"8000": 41231, "22": 10341, "bad": "nope"},
    "ports": ["8000/http", "8000/tcp"],
    "costPerHr": 2.79,
    "machineId": "m1",
    "machine": {"gpuType": {"displayName": "H100 80GB HBM3"}},
    "networkVolume": {"id": "vol123", "dataCenterId": "EU-RO-1"},
    "lastStatusChange": "Rented by User",
}


def test_pod_state_parses_the_documented_response() -> None:
    state = PodState.from_api(PAYLOAD)
    assert state.id == "abc123"
    assert state.is_running
    assert state.has_network
    assert state.port_mappings == {8000: 41231, 22: 10341}
    assert state.gpu_display_name == "H100 80GB HBM3"
    assert state.network_volume_id == "vol123"
    assert state.cost_per_hr == pytest.approx(2.79)


def test_a_freshly_created_pod_has_no_network_yet() -> None:
    state = PodState.from_api({"id": "abc123", "desiredStatus": "RUNNING", "portMappings": None})
    assert state.is_running
    assert not state.has_network
    assert state.tcp_url(8000) is None


# -- url choice ---------------------------------------------------------
def test_direct_tcp_is_preferred_and_carries_no_time_ceiling() -> None:
    endpoint = choose_access_url(PodState.from_api(PAYLOAD), worker_port=8000)
    assert endpoint.kind == "tcp"
    assert endpoint.base_url == "http://213.173.99.7:41231"
    assert endpoint.max_request_seconds is None
    assert not endpoint.is_proxied


def test_proxy_is_the_fallback_and_says_so() -> None:
    without_mapping = PodState.from_api({**PAYLOAD, "portMappings": {"22": 10341}})
    endpoint = choose_access_url(without_mapping, worker_port=8000)
    assert endpoint.kind == "proxy"
    assert endpoint.base_url == "https://abc123-8000.proxy.runpod.net"
    assert endpoint.max_request_seconds == PROXY_TIMEOUT_SECONDS
    assert "524" in endpoint.note or "100s" in endpoint.note


def test_the_proxy_can_be_forced_even_when_tcp_exists() -> None:
    endpoint = choose_access_url(
        PodState.from_api(PAYLOAD), worker_port=8000, prefer_direct_tcp=False
    )
    assert endpoint.kind == "proxy"


def test_default_ports_follow_the_requested_exposure() -> None:
    assert [p.render() for p in default_ports(8000, expose="http")] == ["8000/http"]
    assert [p.render() for p in default_ports(8000, expose="tcp")] == ["8000/tcp"]
    assert [p.render() for p in default_ports(8000, expose="both")] == ["8000/http", "8000/tcp"]
    with pytest.raises(SpecError):
        default_ports(8000, expose="carrier-pigeon")
