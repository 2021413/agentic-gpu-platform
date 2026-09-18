"""The command line, end to end against a mocked RunPod. No network, no GPU.

Every test runs under ``respx.mock``, so a request this code was not supposed
to make fails loudly instead of reaching the real control plane.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest
import respx

from runpod_deployer import cli

KEY = "rpa_TESTKEY0123456789ABCDEF"
ENV = {"RUNPOD_API_KEY": KEY, "HF_TOKEN": "hf_secret_value"}
REST = "https://rest.runpod.io/v1"
IMAGE = "registry.example/gpu-worker:1.0.0"
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

POD = {
    "id": "abc123",
    "name": "worker-1",
    "desiredStatus": "RUNNING",
    "image": IMAGE,
    "publicIp": "213.173.99.7",
    "portMappings": {"8000": 41231},
    "machine": {"gpuType": {"displayName": "H100 80GB HBM3"}},
    "costPerHr": 2.79,
}
POD_PROXY_ONLY = {**POD, "publicIp": None, "portMappings": {}}
COMPLETION = {
    "object": "chat.completion",
    "model": MODEL,
    "choices": [
        {
            "message": {"role": "assistant", "content": "int add(int a,int b){return a+b;}"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 20},
}


class Streams:
    def __init__(self) -> None:
        self.out = io.StringIO()
        self.err = io.StringIO()

    @property
    def text(self) -> str:
        return self.out.getvalue() + self.err.getvalue()


def run(
    argv: list[str], *, env: dict[str, str] | None = None, answer: str = ""
) -> tuple[int, Streams]:
    streams = Streams()
    code = cli.main(
        argv,
        environ=ENV if env is None else env,
        prompt=lambda _message: answer,
        stdout=streams.out,
        stderr=streams.err,
    )
    return code, streams


def mock_pod_routes(pod: dict[str, object] = POD) -> None:
    respx.get(f"{REST}/networkvolumes/vol123").mock(
        return_value=httpx.Response(
            200, json={"id": "vol123", "name": "weights", "size": 200, "dataCenterId": "EU-RO-1"}
        )
    )
    respx.post(f"{REST}/pods").mock(return_value=httpx.Response(201, json=pod))
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(200, json=pod))


DEPLOY_ARGS = [
    "deploy",
    "--image",
    IMAGE,
    "--gpu-type",
    "NVIDIA H100 80GB HBM3",
    "--network-volume",
    "vol123",
]


# -- credentials --------------------------------------------------------
@respx.mock
def test_a_missing_key_exits_with_the_credential_code() -> None:
    code, streams = run(["status", "abc123"], env={})
    assert code == cli.EXIT_CREDENTIALS
    assert "RUNPOD_API_KEY" in streams.err.getvalue()
    assert not respx.calls


@respx.mock
def test_a_rejected_key_also_exits_with_the_credential_code() -> None:
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(401, json={"error": "no"}))
    code, streams = run(["status", "abc123"])
    assert code == cli.EXIT_CREDENTIALS
    assert KEY not in streams.text


# -- dry run ------------------------------------------------------------
@respx.mock
def test_dry_run_prints_the_body_with_secrets_masked_and_creates_nothing() -> None:
    code, streams = run([*DEPLOY_ARGS, "--dry-run"])
    assert code == cli.EXIT_OK
    assert not respx.calls
    body = json.loads(
        streams.out.getvalue()[streams.out.getvalue().index("{") :].rsplit("}", 1)[0] + "}"
    )
    assert body["env"]["HF_TOKEN"] == "***"
    assert "hf_secret_value" not in streams.text
    assert body["ports"] == ["8000/http", "8000/tcp"]


@respx.mock
def test_deploy_without_an_image_is_a_clear_refusal() -> None:
    code, streams = run(["deploy", "--dry-run"], env={"RUNPOD_API_KEY": KEY})
    assert code == cli.EXIT_API
    assert "WORKER_IMAGE" in streams.err.getvalue()


# -- deploy -------------------------------------------------------------
@respx.mock
def test_deploy_reports_timings_and_the_direct_tcp_url() -> None:
    mock_pod_routes()
    respx.get("http://213.173.99.7:41231/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": MODEL}]})
    )
    respx.post("http://213.173.99.7:41231/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )
    code, streams = run([*DEPLOY_ARGS, "--json"])
    out = streams.out.getvalue()

    assert code == cli.EXIT_OK
    assert "http://213.173.99.7:41231" in out
    assert "smoke test passed" in out
    assert "cold_start_seconds" in out
    assert "WARNING" not in out, "a direct TCP endpoint has no proxy caveat"
    # the data center of the volume constrains the Pod
    created = json.loads(respx.calls[1].request.content)
    assert created["dataCenterIds"] == ["EU-RO-1"]


@respx.mock
def test_deploy_through_the_proxy_warns_about_the_100_second_ceiling() -> None:
    mock_pod_routes(POD_PROXY_ONLY)
    respx.get("https://abc123-8000.proxy.runpod.net/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": MODEL}]})
    )
    respx.post("https://abc123-8000.proxy.runpod.net/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )
    code, streams = run([*DEPLOY_ARGS, "--expose", "http"])
    out = streams.out.getvalue()

    assert code == cli.EXIT_OK
    assert "https://abc123-8000.proxy.runpod.net" in out
    assert "WARNING" in out
    assert "100s" in out or "100 s" in out
    assert "524" in out


@respx.mock
def test_a_worker_that_never_serves_the_model_exits_not_ready_and_keeps_the_pod() -> None:
    mock_pod_routes()
    respx.get("http://213.173.99.7:41231/v1/models").mock(return_value=httpx.Response(503))
    code, streams = run([*DEPLOY_ARGS, "--ready-timeout", "0.01", "--poll", "0.01"])

    assert code == cli.EXIT_NOT_READY
    assert "still running and billing" in streams.err.getvalue()
    assert not any(call.request.method == "DELETE" for call in respx.calls)


@respx.mock
def test_a_failing_smoke_test_exits_with_its_own_code() -> None:
    mock_pod_routes()
    respx.get("http://213.173.99.7:41231/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": MODEL}]})
    )
    respx.post("http://213.173.99.7:41231/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={**COMPLETION, "choices": []})
    )
    code, streams = run(DEPLOY_ARGS)
    assert code == cli.EXIT_SMOKE_FAILED
    assert "smoke test FAILED" in streams.out.getvalue()


@respx.mock
def test_a_pod_that_never_comes_up_exits_with_the_timeout_code() -> None:
    respx.get(f"{REST}/networkvolumes/vol123").mock(
        return_value=httpx.Response(
            200, json={"id": "vol123", "name": "w", "size": 200, "dataCenterId": "EU-RO-1"}
        )
    )
    respx.post(f"{REST}/pods").mock(return_value=httpx.Response(201, json={"id": "abc123"}))
    respx.get(f"{REST}/pods/abc123").mock(
        return_value=httpx.Response(200, json={"id": "abc123", "desiredStatus": "PENDING"})
    )
    code, streams = run([*DEPLOY_ARGS, "--pod-timeout", "0.01", "--poll", "0.01"])
    assert code == cli.EXIT_POD_TIMEOUT
    assert "never came up" in streams.err.getvalue()


@respx.mock
def test_deploy_without_a_volume_says_what_will_be_lost() -> None:
    respx.post(f"{REST}/pods").mock(return_value=httpx.Response(201, json=POD))
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(200, json=POD))
    respx.get("http://213.173.99.7:41231/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": MODEL}]})
    )
    respx.post("http://213.173.99.7:41231/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )
    code, streams = run(["deploy", "--image", IMAGE])
    assert code == cli.EXIT_OK
    assert "31 GB" in streams.out.getvalue()


# -- status and smoke ---------------------------------------------------
@respx.mock
def test_status_reports_the_url_it_would_use() -> None:
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(200, json=POD))
    code, streams = run(["status", "abc123", "--json"])
    assert code == cli.EXIT_OK
    payload = json.loads(streams.out.getvalue()[streams.out.getvalue().index("{") :])
    assert payload["url_kind"] == "tcp"
    assert payload["max_request_seconds"] is None


@respx.mock
def test_status_through_the_proxy_reports_the_ceiling() -> None:
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(200, json=POD_PROXY_ONLY))
    code, streams = run(["status", "abc123", "--json"])
    payload = json.loads(streams.out.getvalue()[streams.out.getvalue().index("{") :])
    assert code == cli.EXIT_OK
    assert payload["url_kind"] == "proxy"
    assert payload["max_request_seconds"] == 100


@respx.mock
def test_smoke_test_command_needs_a_model_name() -> None:
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(200, json=POD))
    code, streams = run(["smoke-test", "abc123"], env={"RUNPOD_API_KEY": KEY})
    assert code == cli.EXIT_USAGE
    assert "--model" in streams.err.getvalue()


@respx.mock
def test_smoke_test_command_passes_when_the_model_answers() -> None:
    respx.get(f"{REST}/pods/abc123").mock(return_value=httpx.Response(200, json=POD))
    respx.post("http://213.173.99.7:41231/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )
    code, streams = run(["smoke-test", "abc123", "--model", MODEL])
    assert code == cli.EXIT_OK
    assert "smoke test passed" in streams.out.getvalue()


# -- destroy ------------------------------------------------------------
@respx.mock
def test_destroy_refuses_without_confirmation() -> None:
    route = respx.delete(f"{REST}/pods/abc123").mock(return_value=httpx.Response(204))
    code, streams = run(["destroy", "abc123"], answer="")
    assert code == cli.EXIT_ABORTED
    assert not route.called
    assert "nothing was terminated" in streams.err.getvalue()


@respx.mock
def test_destroy_refuses_when_the_answer_is_not_the_pod_id() -> None:
    route = respx.delete(f"{REST}/pods/abc123").mock(return_value=httpx.Response(204))
    code, _ = run(["destroy", "abc123"], answer="yes")
    assert code == cli.EXIT_ABORTED
    assert not route.called


@respx.mock
def test_destroy_proceeds_when_the_pod_id_is_typed_back() -> None:
    route = respx.delete(f"{REST}/pods/abc123").mock(return_value=httpx.Response(204))
    code, _ = run(["destroy", "abc123"], answer="abc123")
    assert code == cli.EXIT_OK
    assert route.called


@respx.mock
def test_destroy_yes_skips_the_prompt() -> None:
    route = respx.delete(f"{REST}/pods/abc123").mock(return_value=httpx.Response(204))
    code, streams = run(["destroy", "abc123", "--yes"])
    assert code == cli.EXIT_OK
    assert route.called
    assert "terminated" in streams.out.getvalue()


# -- gpu types ----------------------------------------------------------
@respx.mock
def test_gpu_types_lists_and_filters() -> None:
    respx.post(host="api.runpod.io").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "gpuTypes": [
                        {
                            "id": "NVIDIA H100 80GB HBM3",
                            "memoryInGb": 80,
                            "secureCloud": True,
                            "securePrice": 2.79,
                        },
                        {"id": "NVIDIA GeForce RTX 4090", "memoryInGb": 24, "communityCloud": True},
                    ]
                }
            },
        )
    )
    code, streams = run(["gpu-types", "--min-memory", "48"])
    out = streams.out.getvalue()
    assert code == cli.EXIT_OK
    assert "H100" in out
    assert "4090" not in out
    assert "1 of 2 types shown" in out


# -- usage --------------------------------------------------------------
def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == cli.EXIT_USAGE
