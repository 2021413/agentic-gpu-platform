"""The RunPod client, against a mocked transport. No network, no GPU.

The three things worth proving here are the ones that cost money or leak a
credential when they are wrong: the exact body sent to create a Pod, the fact
that a rejected key is never retried, and the fact that nothing which leaves
this module can carry the API key.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from runpod_deployer import client as api
from runpod_deployer.models import PodSpec, SecretValue, WorkerSettings, default_ports

KEY = "rpa_TESTKEY0123456789ABCDEF"
ENV = {"RUNPOD_API_KEY": KEY}
REST = "https://rest.runpod.io/v1"
IMAGE = "registry.example/gpu-worker:1.0.0"

POD_RUNNING = {
    "id": "abc123",
    "name": "worker-1",
    "desiredStatus": "RUNNING",
    "image": IMAGE,
    "publicIp": "213.173.99.7",
    "portMappings": {"8000": 41231},
    "machine": {"gpuType": {"displayName": "H100 80GB HBM3"}},
}


class Clock:
    """A sleep that records instead of waiting."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def make_client(environ: dict[str, str] | None = None, **kwargs: object) -> api.RunPodClient:
    return api.RunPodClient(environ=environ if environ is not None else ENV, **kwargs)  # type: ignore[arg-type]


def a_spec() -> PodSpec:
    return PodSpec(
        image_name=IMAGE,
        name="worker-1",
        gpu_type_ids=("NVIDIA H100 80GB HBM3",),
        network_volume_id="vol123",
        ports=default_ports(8000, expose="both"),
        worker=WorkerSettings(hf_token=SecretValue("hf_secret_value")),
    )


# -- credential ---------------------------------------------------------
def test_the_key_comes_from_the_environment_and_nowhere_else() -> None:
    with pytest.raises(api.MissingApiKeyError) as excinfo:
        make_client({})
    rendered = excinfo.value.render()
    assert "RUNPOD_API_KEY" in rendered
    assert "export RUNPOD_API_KEY" in rendered
    assert "never accepts the key as an argument" in rendered


def test_a_blank_key_is_treated_as_absent() -> None:
    with pytest.raises(api.MissingApiKeyError):
        make_client({"RUNPOD_API_KEY": "   "})


def test_the_client_never_reprs_its_key() -> None:
    with make_client() as client:
        assert KEY not in repr(client)
        assert KEY not in str(client)
        assert "<redacted>" in repr(client)


# -- redaction ----------------------------------------------------------
def test_redact_masks_known_secrets_shapes_and_headers() -> None:
    text = (
        f"POST https://api.runpod.io/graphql?api_key={KEY} failed; "
        f"header was Authorization: Bearer {KEY}; other key rpa_OTHERKEY12345678"
    )
    cleaned = api.redact(text, secrets=(KEY,))
    assert KEY not in cleaned
    assert "rpa_OTHERKEY12345678" not in cleaned
    assert "api_key=***" in cleaned
    assert "Bearer ***" in cleaned


def test_redact_leaves_ordinary_text_alone() -> None:
    assert api.redact("pod abc123 is RUNNING") == "pod abc123 is RUNNING"


@respx.mock
def test_an_error_body_echoing_the_key_is_masked_before_it_propagates() -> None:
    respx.post(f"{REST}/pods").mock(
        return_value=httpx.Response(
            400,
            json={"error": f"invalid request for key {KEY} on https://x?api_key={KEY}"},
        )
    )
    with make_client() as client, pytest.raises(api.RunPodError) as excinfo:
        client.create_pod(a_spec())
    message = excinfo.value.render()
    assert KEY not in message
    assert "***" in message


@respx.mock
def test_a_graphql_error_echoing_the_key_is_masked() -> None:
    respx.post(host="api.runpod.io").mock(
        return_value=httpx.Response(200, json={"errors": [{"message": f"bad key {KEY}"}]})
    )
    with make_client() as client, pytest.raises(api.RunPodError) as excinfo:
        client.list_gpu_types()
    assert KEY not in str(excinfo.value)


# -- create pod ---------------------------------------------------------
@respx.mock
def test_create_pod_sends_the_documented_body_with_bearer_auth() -> None:
    route = respx.post(f"{REST}/pods").mock(return_value=httpx.Response(201, json=POD_RUNNING))
    with make_client() as client:
        state = client.create_pod(a_spec())

    assert state.id == "abc123"
    assert state.port_mappings == {8000: 41231}
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == a_spec().to_create_body()


@respx.mock
def test_create_pod_without_an_id_is_an_error_not_a_pod() -> None:
    respx.post(f"{REST}/pods").mock(return_value=httpx.Response(201, json={"status": "queued"}))
    with make_client() as client, pytest.raises(api.ApiError, match="no id"):
        client.create_pod(a_spec())


# -- reading and terminating -------------------------------------------
@respx.mock
def test_get_pod_asks_for_the_machine_block() -> None:
    route = respx.get(f"{REST}/pods/abc123").mock(
        return_value=httpx.Response(200, json=POD_RUNNING)
    )
    with make_client() as client:
        state = client.get_pod("abc123")
    assert state.gpu_display_name == "H100 80GB HBM3"
    assert route.calls.last.request.url.params["includeMachine"] == "true"


@respx.mock
def test_terminate_accepts_the_documented_204() -> None:
    route = respx.delete(f"{REST}/pods/abc123").mock(return_value=httpx.Response(204))
    with make_client() as client:
        assert client.terminate_pod("abc123") is None
    assert route.called


@respx.mock
def test_a_missing_pod_is_named_as_such() -> None:
    respx.get(f"{REST}/pods/ghost").mock(return_value=httpx.Response(404, json={"error": "nope"}))
    with make_client() as client, pytest.raises(api.PodNotFoundError):
        client.get_pod("ghost")


@respx.mock
def test_a_missing_volume_points_at_the_storage_console() -> None:
    respx.get(f"{REST}/networkvolumes/volX").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    with make_client() as client, pytest.raises(api.VolumeNotFoundError) as excinfo:
        client.get_network_volume("volX")
    assert "storage" in excinfo.value.render()


@respx.mock
def test_a_volume_carries_its_data_center() -> None:
    respx.get(f"{REST}/networkvolumes/vol123").mock(
        return_value=httpx.Response(
            200, json={"id": "vol123", "name": "weights", "size": 200, "dataCenterId": "EU-RO-1"}
        )
    )
    with make_client() as client:
        volume = client.get_network_volume("vol123")
    assert volume.data_center_id == "EU-RO-1"
    assert volume.size == 200


# -- error translation --------------------------------------------------
@respx.mock
def test_no_capacity_is_reported_as_a_gpu_problem_with_a_next_step() -> None:
    respx.post(f"{REST}/pods").mock(
        return_value=httpx.Response(400, json={"error": "no instances available for gpu type"})
    )
    with make_client() as client, pytest.raises(api.GpuUnavailableError) as excinfo:
        client.create_pod(a_spec())
    assert "gpu-types" in excinfo.value.render()


@respx.mock
def test_a_rejected_volume_explains_the_data_center_constraint() -> None:
    respx.post(f"{REST}/pods").mock(
        return_value=httpx.Response(400, json={"error": "network volume is not available here"})
    )
    with make_client() as client, pytest.raises(api.VolumeNotFoundError) as excinfo:
        client.create_pod(a_spec())
    assert "data center" in excinfo.value.render()


@respx.mock
def test_payment_required_is_a_quota_error() -> None:
    respx.post(f"{REST}/pods").mock(return_value=httpx.Response(402, json={"error": "no funds"}))
    with make_client() as client, pytest.raises(api.QuotaError):
        client.create_pod(a_spec())


# -- retries ------------------------------------------------------------
@respx.mock
def test_a_rejected_key_is_never_retried() -> None:
    route = respx.post(f"{REST}/pods").mock(
        return_value=httpx.Response(401, json={"error": "Unauthorized"})
    )
    sleep = Clock()
    with make_client(sleep=sleep) as client, pytest.raises(api.AuthenticationError) as excinfo:
        client.create_pod(a_spec())

    assert route.call_count == 1, "an invalid key does not become valid by asking again"
    assert sleep.slept == []
    assert "console.runpod.io" in excinfo.value.render()


@respx.mock
def test_a_forbidden_key_is_never_retried_either() -> None:
    route = respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(403))
    with make_client(sleep=Clock()) as client, pytest.raises(api.AuthenticationError):
        client.get_pod("abc123")
    assert route.call_count == 1


@respx.mock
def test_a_transient_failure_is_retried_then_succeeds() -> None:
    route = respx.post(f"{REST}/pods").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream busy"}),
            httpx.Response(201, json=POD_RUNNING),
        ]
    )
    sleep = Clock()
    with make_client(sleep=sleep) as client:
        state = client.create_pod(a_spec())
    assert state.id == "abc123"
    assert route.call_count == 2
    assert sleep.slept == [2.0]


@respx.mock
def test_retries_are_bounded_and_the_last_error_is_raised() -> None:
    route = respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(502))
    sleep = Clock()
    with make_client(sleep=sleep) as client, pytest.raises(api.ServiceError):
        client.get_pod("abc123")
    assert route.call_count == 3
    assert sleep.slept == [2.0, 4.0]


@respx.mock
def test_a_connection_failure_is_retried_and_then_reported_as_transport() -> None:
    route = respx.get(f"{REST}/pods/abc123").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with make_client(sleep=Clock()) as client, pytest.raises(api.TransportError) as excinfo:
        client.get_pod("abc123")
    assert route.call_count == 3
    assert "may still have been created" in excinfo.value.render()


# -- gpu types ----------------------------------------------------------
@respx.mock
def test_gpu_types_come_from_graphql_and_are_parsed() -> None:
    route = respx.post(host="api.runpod.io").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "gpuTypes": [
                        {
                            "id": "NVIDIA H100 80GB HBM3",
                            "displayName": "H100 80GB HBM3",
                            "memoryInGb": 80,
                            "secureCloud": True,
                            "communityCloud": False,
                            "securePrice": 2.79,
                            "communityPrice": None,
                        },
                        "not a dict",
                    ]
                }
            },
        )
    )
    with make_client() as client:
        types = client.list_gpu_types()

    assert [gpu.id for gpu in types] == ["NVIDIA H100 80GB HBM3"]
    assert types[0].memory_in_gb == 80
    assert types[0].secure_price == pytest.approx(2.79)
    body = route.calls.last.request.content.decode()
    assert "gpuTypes" in body
    assert "memoryInGb" in body


# -- worker-side probes -------------------------------------------------
BASE = "http://213.173.99.7:41231"
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
COMPLETION = {
    "object": "chat.completion",
    "model": MODEL,
    "choices": [
        {
            "message": {"role": "assistant", "content": "int add(int a, int b){return a+b;}"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 20},
}


@respx.mock
def test_readiness_waits_for_the_model_to_be_listed_not_merely_a_200() -> None:
    respx.get(f"{BASE}/v1/models").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"data": [{"id": "some-other-model"}]}),
            httpx.Response(200, json={"data": [{"id": MODEL}]}),
        ]
    )
    ticks = iter([0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    report = api.wait_until_ready(
        BASE,
        MODEL,
        timeout_seconds=600.0,
        poll_seconds=5.0,
        now=lambda: next(ticks),
        sleep=Clock(),
    )
    assert report.ready
    assert MODEL in report.models


@respx.mock
def test_readiness_gives_up_and_says_what_it_saw() -> None:
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    report = api.wait_until_ready(BASE, MODEL, timeout_seconds=1.0, poll_seconds=0.1, sleep=Clock())
    assert not report.ready
    assert "expected" in (report.detail or "")


@respx.mock
def test_smoke_test_validates_the_whole_answer() -> None:
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )
    report = api.smoke_test(BASE, MODEL, api_key=SecretValue("sk-worker"))
    assert report.passed
    assert report.completion_tokens == 20
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-worker"


@respx.mock
def test_an_empty_answer_is_a_failure_even_with_http_200() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                **COMPLETION,
                "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
            },
        )
    )
    report = api.smoke_test(BASE, MODEL)
    assert not report.passed
    assert "no output" in (report.detail or "")


@respx.mock
def test_a_524_is_reported_as_the_proxy_giving_up() -> None:
    respx.post("https://abc123-8000.proxy.runpod.net/v1/chat/completions").mock(
        return_value=httpx.Response(524, text="error code: 524")
    )
    report = api.smoke_test("https://abc123-8000.proxy.runpod.net", MODEL)
    assert not report.passed
    assert "100s" in (report.detail or "")


@respx.mock
def test_first_byte_is_measured_from_the_stream() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text='data: {"choices":[]}\n\ndata: [DONE]\n\n')
    )
    ticks = iter([0.0, 1.5, 3.0])
    report = api.measure_first_byte(BASE, MODEL, now=lambda: next(ticks))
    assert report.ok
    assert report.seconds == pytest.approx(1.5)
