"""Typed description of a RunPod Pod: what we ask for, and what we got back.

No socket is opened from this module. Everything here is pure: a spec turns
into the exact JSON body the REST API expects, a JSON response turns into a
``PodState``, and a state plus a port turns into the URL an operator should
use. That purity is what lets the unit tests assert the *exact* request body
without a network, and what keeps the interesting decisions (which URL, which
variables, which of them are secret) testable one by one.

Field names follow the RunPod REST schema verbatim -- ``PodCreateInput`` and
``Pod`` in https://rest.runpod.io/v1/openapi.json. Environment variable names
follow ``src/worker/config.py``: this module is the other half of that contract
and the two must be read together. The names are repeated here rather than
imported because the deployer runs on an operator laptop, where ``src/`` is not
necessarily importable, while the worker runs in the container.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

__all__ = [
    "PROXY_TIMEOUT_SECONDS",
    "AccessEndpoint",
    "ExposedPort",
    "GpuType",
    "NetworkVolume",
    "PodEnvironment",
    "PodSpec",
    "PodState",
    "SecretValue",
    "SpecError",
    "WorkerSettings",
    "build_pod_environment",
    "choose_access_url",
]

# The Cloudflare proxy in front of ``*.proxy.runpod.net`` closes a connection
# that has not answered within this many seconds, with a 524. Anything slower
# than this -- a long generation, a cold model load behind the same port -- has
# to go over a direct TCP mapping instead.
# https://docs.runpod.io/pods/configuration/expose-ports
PROXY_TIMEOUT_SECONDS = 100

PROXY_WARNING = (
    "this is the Cloudflare HTTP proxy: any single request that takes longer "
    f"than {PROXY_TIMEOUT_SECONDS}s is cut with a 524, whatever the server does. "
    "Long generations must either stream early tokens, be split, or go through a "
    "direct TCP port (expose the worker port as tcp)."
)

TCP_NOTE = (
    "direct TCP, no Cloudflare in front: no 100s ceiling. The external port is "
    "re-assigned on every Pod reset, so an orchestrator must re-read it rather "
    "than cache it."
)

# Mirrors src/worker/config.py. Changing a default here without changing it
# there produces a Pod that boots with settings nobody chose.
DEFAULT_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
DEFAULT_PERSISTENT_ROOT = "/runpod-volume"
DEFAULT_WORKER_PORT = 8000
DEFAULT_MIN_FREE_DISK_GB = 60.0

MAX_TCP_PORT = 65535
SYMMETRIC_PORT_FLOOR = 70000  # ports requested above this ask for a 1:1 mapping


class SpecError(ValueError):
    """A Pod description that cannot work, caught before any money is spent."""


class SecretValue:
    """A string that refuses to render itself.

    Mirrors ``worker.config.Secret``. A deployer prints its plan, its request
    body and its errors constantly; a token must not be able to ride along in
    any of them by accident. Reading the value is a call you can grep for.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str | None = None) -> None:
        self._value = value or ""

    def __bool__(self) -> bool:
        return bool(self._value)

    def reveal(self) -> str:
        """The bytes themselves. One call site per purpose, on purpose."""
        return self._value

    def __repr__(self) -> str:
        return "SecretValue(set)" if self._value else "SecretValue(unset)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretValue):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("SecretValue", self._value))


@dataclass(frozen=True, slots=True)
class ExposedPort:
    """One entry of ``ports``, formatted ``[number]/[protocol]``."""

    number: int
    protocol: Literal["http", "tcp"] = "http"

    def __post_init__(self) -> None:
        if not 1 <= self.number <= SYMMETRIC_PORT_FLOOR + MAX_TCP_PORT:
            raise SpecError(f"port {self.number} is not a usable port number")
        if self.protocol not in ("http", "tcp"):
            raise SpecError(f"protocol must be http or tcp, got {self.protocol!r}")

    def render(self) -> str:
        return f"{self.number}/{self.protocol}"


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """What the container reads from its environment.

    One field per variable of ``src/worker/config.py`` that the deployer is
    entitled to set. Secrets are last and typed differently so that they cannot
    be printed by mistake.
    """

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    served_model_name: str | None = None
    persistent_root: str = DEFAULT_PERSISTENT_ROOT
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB
    port: int = DEFAULT_WORKER_PORT
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    auto_tensor_parallel: bool = False
    extra_args: tuple[str, ...] = ()
    readiness_timeout_seconds: float = 1800.0
    hf_hub_disable_xet: bool | None = None

    hf_token: SecretValue = field(default_factory=SecretValue)
    vllm_api_key: SecretValue = field(default_factory=SecretValue)

    @property
    def public_model_name(self) -> str:
        """What a client must send as ``model``; mirrors the worker's rule."""
        return self.served_model_name or self.model_id


@dataclass(frozen=True, slots=True)
class PodEnvironment:
    """The environment of a Pod, split by whether it may be shown.

    ``public`` can go in a log, a plan printout or a bug report. ``secret``
    may only ever be revealed into the request body that creates the Pod.
    """

    public: dict[str, str]
    secret: dict[str, SecretValue]

    def as_api_env(self) -> dict[str, str]:
        """The merged mapping sent to RunPod. The only place values are read."""
        merged = dict(self.public)
        for name, secret in self.secret.items():
            if secret:
                merged[name] = secret.reveal()
        return merged

    def redacted(self) -> dict[str, str]:
        """The same mapping, safe to print."""
        merged = dict(self.public)
        for name, secret in self.secret.items():
            if secret:
                merged[name] = "***"
        return merged

    def secret_names(self) -> tuple[str, ...]:
        return tuple(name for name, secret in self.secret.items() if secret)


def build_pod_environment(settings: WorkerSettings) -> PodEnvironment:
    """Translate worker settings into the Pod's environment.

    Only variables the operator actually chose are emitted: sending the whole
    contract with its defaults would freeze today's defaults into every Pod and
    hide a later change to ``config.py`` behind an explicit value.
    """
    public: dict[str, str] = {
        "MODEL_ID": settings.model_id,
        "PERSISTENT_ROOT": settings.persistent_root,
        "PORT": str(settings.port),
        "MAX_MODEL_LEN": str(settings.max_model_len),
        "GPU_MEMORY_UTILIZATION": f"{settings.gpu_memory_utilization:g}",
        "TENSOR_PARALLEL_SIZE": str(settings.tensor_parallel_size),
        "MIN_FREE_DISK_GB": f"{settings.min_free_disk_gb:g}",
        "READINESS_TIMEOUT_SECONDS": f"{settings.readiness_timeout_seconds:g}",
    }
    if settings.model_revision:
        public["MODEL_REVISION"] = settings.model_revision
    if settings.served_model_name:
        public["SERVED_MODEL_NAME"] = settings.served_model_name
    if settings.auto_tensor_parallel:
        public["AUTO_TENSOR_PARALLEL"] = "1"
    if settings.extra_args:
        public["VLLM_EXTRA_ARGS"] = shlex.join(settings.extra_args)
    if settings.hf_hub_disable_xet is not None:
        public["HF_HUB_DISABLE_XET"] = "1" if settings.hf_hub_disable_xet else "0"

    secret: dict[str, SecretValue] = {}
    if settings.hf_token:
        secret["HF_TOKEN"] = settings.hf_token
    if settings.vllm_api_key:
        secret["VLLM_API_KEY"] = settings.vllm_api_key
    return PodEnvironment(public=public, secret=secret)


@dataclass(frozen=True, slots=True)
class PodSpec:
    """Everything needed to create one Pod, validated before it costs anything.

    The validation is the point: a mount path that does not match
    ``PERSISTENT_ROOT`` silently sends a 31 GB download to the container disk,
    which is wiped on restart, and the mistake is only visible half an hour and
    one full download later.
    """

    image_name: str
    gpu_type_ids: tuple[str, ...] = ()
    name: str = "gpu-worker"
    gpu_count: int = 1
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    data_center_ids: tuple[str, ...] = ()
    container_disk_in_gb: int = 60
    volume_in_gb: int = 0
    network_volume_id: str | None = None
    volume_mount_path: str = DEFAULT_PERSISTENT_ROOT
    ports: tuple[ExposedPort, ...] = ()
    support_public_ip: bool = True
    global_networking: bool = False
    interruptible: bool = False
    worker: WorkerSettings = field(default_factory=WorkerSettings)

    def __post_init__(self) -> None:
        if not self.image_name.strip():
            raise SpecError("imageName must not be empty")
        if self.gpu_count < 1:
            raise SpecError("gpuCount must be at least 1")
        if self.cloud_type not in ("SECURE", "COMMUNITY"):
            raise SpecError(f"cloudType must be SECURE or COMMUNITY, got {self.cloud_type!r}")
        if self.container_disk_in_gb < 1:
            raise SpecError("containerDiskInGb must be at least 1")
        if not self.ports:
            raise SpecError("a Pod with no exposed port cannot be reached")
        if self.worker.port not in {port.number for port in self.ports}:
            raise SpecError(
                f"the worker listens on {self.worker.port} but that port is not exposed "
                f"({', '.join(port.render() for port in self.ports)})"
            )
        if self.network_volume_id and self.volume_mount_path != self.worker.persistent_root:
            raise SpecError(
                f"the network volume is mounted at {self.volume_mount_path} but the worker "
                f"writes to {self.worker.persistent_root}: the model would land on the "
                "container disk and be lost on the next restart"
            )

    # -- derived ---------------------------------------------------------
    @property
    def http_ports(self) -> tuple[int, ...]:
        return tuple(port.number for port in self.ports if port.protocol == "http")

    @property
    def tcp_ports(self) -> tuple[int, ...]:
        return tuple(port.number for port in self.ports if port.protocol == "tcp")

    def environment(self) -> PodEnvironment:
        return build_pod_environment(self.worker)

    def with_worker(self, **changes: Any) -> PodSpec:
        return replace(self, worker=replace(self.worker, **changes))

    def to_create_body(self) -> dict[str, Any]:
        """The exact JSON body of ``POST https://rest.runpod.io/v1/pods``.

        Optional keys are omitted rather than sent as null: the API fills in
        documented defaults, and an explicit null is a different thing from an
        absent field for several of them.
        """
        body: dict[str, Any] = {
            "name": self.name,
            "imageName": self.image_name,
            "computeType": "GPU",
            "cloudType": self.cloud_type,
            "gpuCount": self.gpu_count,
            "containerDiskInGb": self.container_disk_in_gb,
            "volumeMountPath": self.volume_mount_path,
            "ports": [port.render() for port in self.ports],
            "env": self.environment().as_api_env(),
            "interruptible": self.interruptible,
        }
        if self.gpu_type_ids:
            body["gpuTypeIds"] = list(self.gpu_type_ids)
        if self.data_center_ids:
            body["dataCenterIds"] = list(self.data_center_ids)
        if self.network_volume_id:
            # A network volume replaces the Pod volume, so volumeInGb would be
            # ignored; sending it anyway only invites confusion at review time.
            body["networkVolumeId"] = self.network_volume_id
        elif self.volume_in_gb:
            body["volumeInGb"] = self.volume_in_gb
        if self.cloud_type == "COMMUNITY" and self.tcp_ports:
            # Community Cloud machines do not all carry a public IP; without
            # this, a tcp port simply never gets a mapping.
            body["supportPublicIp"] = self.support_public_ip
        if self.global_networking:
            body["globalNetworking"] = True
        return body

    def describe(self) -> list[str]:
        """Operator-facing plan. Contains no secret, by construction."""
        env = self.environment()
        return [
            f"name              {self.name}",
            f"image             {self.image_name}",
            f"gpu               {', '.join(self.gpu_type_ids) or 'any'} x{self.gpu_count}",
            f"cloud             {self.cloud_type}",
            f"data centers      {', '.join(self.data_center_ids) or 'any'}",
            f"container disk    {self.container_disk_in_gb} GB",
            f"network volume    {self.network_volume_id or 'NONE (nothing persists)'}"
            f" at {self.volume_mount_path}",
            f"ports             {', '.join(port.render() for port in self.ports)}",
            f"worker port       {self.worker.port}",
            f"model             {self.worker.public_model_name}",
            f"secrets injected  {', '.join(env.secret_names()) or 'none'}",
        ]


@dataclass(frozen=True, slots=True)
class PodState:
    """A Pod as RunPod reports it.

    ``desired_status`` is what the API calls the lifecycle state; the network
    details (``public_ip``, ``port_mappings``) stay empty while the Pod is still
    being placed, which is what makes them the real signal that it is up.
    """

    id: str
    name: str = ""
    desired_status: str = ""
    image: str = ""
    public_ip: str | None = None
    port_mappings: dict[int, int] = field(default_factory=dict)
    ports: tuple[str, ...] = ()
    cost_per_hr: float | None = None
    machine_id: str = ""
    gpu_display_name: str = ""
    last_status_change: str = ""
    network_volume_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_running(self) -> bool:
        return self.desired_status == "RUNNING"

    @property
    def is_terminated(self) -> bool:
        return self.desired_status == "TERMINATED"

    @property
    def has_network(self) -> bool:
        """True once the placement produced something addressable."""
        return bool(self.public_ip) and bool(self.port_mappings)

    def proxy_url(self, port: int) -> str:
        return f"https://{self.id}-{port}.proxy.runpod.net"

    def tcp_url(self, port: int) -> str | None:
        mapped = self.port_mappings.get(port)
        if not mapped or not self.public_ip:
            return None
        return f"http://{self.public_ip}:{mapped}"

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> PodState:
        machine = payload.get("machine") or {}
        gpu_type = machine.get("gpuType") if isinstance(machine, dict) else None
        network_volume = payload.get("networkVolume") or {}
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            desired_status=str(payload.get("desiredStatus", "")),
            image=str(payload.get("image", "")),
            public_ip=(payload.get("publicIp") or None),
            port_mappings=_port_mappings(payload.get("portMappings")),
            ports=tuple(str(p) for p in (payload.get("ports") or ())),
            cost_per_hr=_optional_float(payload.get("costPerHr")),
            machine_id=str(payload.get("machineId", "")),
            gpu_display_name=str((gpu_type or {}).get("displayName", ""))
            if isinstance(gpu_type, dict)
            else "",
            last_status_change=str(payload.get("lastStatusChange", "")),
            network_volume_id=(
                str(network_volume.get("id"))
                if isinstance(network_volume, dict) and network_volume.get("id")
                else None
            ),
            raw=dict(payload),
        )

    def describe(self) -> list[str]:
        mappings = ", ".join(f"{k}->{v}" for k, v in sorted(self.port_mappings.items()))
        return [
            f"id                {self.id}",
            f"name              {self.name}",
            f"status            {self.desired_status or 'unknown'}",
            f"image             {self.image}",
            f"gpu               {self.gpu_display_name or 'unknown'}",
            f"public ip         {self.public_ip or 'not assigned yet'}",
            f"port mappings     {mappings or 'not assigned yet'}",
            f"network volume    {self.network_volume_id or 'none'}",
            f"cost              {f'{self.cost_per_hr:.3f}/hr' if self.cost_per_hr else 'unknown'}",
            f"last change       {self.last_status_change or 'unknown'}",
        ]


@dataclass(frozen=True, slots=True)
class AccessEndpoint:
    """Where to talk to the worker, and what that choice costs."""

    kind: Literal["tcp", "proxy"]
    base_url: str
    note: str

    @property
    def is_proxied(self) -> bool:
        return self.kind == "proxy"

    @property
    def max_request_seconds(self) -> int | None:
        """The hard ceiling on a single request, if there is one."""
        return PROXY_TIMEOUT_SECONDS if self.is_proxied else None


def choose_access_url(
    state: PodState,
    *,
    worker_port: int,
    prefer_direct_tcp: bool = True,
) -> AccessEndpoint:
    """Pick the URL to hand to the orchestrator.

    Direct TCP wins whenever it exists, because the alternative silently caps
    every request at 100 seconds. The proxy is the fallback, never the default,
    and it is returned with the warning attached rather than as a bare string so
    that no caller can forget to say so.
    """
    if prefer_direct_tcp:
        direct = state.tcp_url(worker_port)
        if direct:
            return AccessEndpoint(kind="tcp", base_url=direct, note=TCP_NOTE)
    return AccessEndpoint(kind="proxy", base_url=state.proxy_url(worker_port), note=PROXY_WARNING)


@dataclass(frozen=True, slots=True)
class GpuType:
    """One entry of the GraphQL ``gpuTypes`` query."""

    id: str
    display_name: str = ""
    memory_in_gb: int = 0
    secure_cloud: bool = False
    community_cloud: bool = False
    secure_price: float | None = None
    community_price: float | None = None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> GpuType:
        return cls(
            id=str(payload.get("id", "")),
            display_name=str(payload.get("displayName", "")),
            memory_in_gb=int(payload.get("memoryInGb") or 0),
            secure_cloud=bool(payload.get("secureCloud")),
            community_cloud=bool(payload.get("communityCloud")),
            secure_price=_optional_float(payload.get("securePrice")),
            community_price=_optional_float(payload.get("communityPrice")),
        )

    def render(self) -> str:
        clouds = ",".join(
            kind
            for kind, flag in (("secure", self.secure_cloud), ("community", self.community_cloud))
            if flag
        )
        price = f"{self.secure_price:.2f}" if self.secure_price is not None else "?"
        return f"{self.id:<34} {self.memory_in_gb:>4} GB  {clouds or 'none':<17} {price}/hr"


@dataclass(frozen=True, slots=True)
class NetworkVolume:
    """One entry of ``GET /networkvolumes``."""

    id: str
    name: str = ""
    size: int = 0
    data_center_id: str = ""

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> NetworkVolume:
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            size=int(payload.get("size") or 0),
            data_center_id=str(payload.get("dataCenterId", "")),
        )


def _port_mappings(raw: object) -> dict[int, int]:
    """``{"8000": 41231}`` as reported becomes ``{8000: 41231}``.

    Entries that are not a pair of integers are dropped rather than raising: a
    Pod that reports one odd mapping is still usable, and a deployer that dies
    on parsing leaves a GPU running.
    """
    if not isinstance(raw, Mapping):
        return {}
    mappings: dict[int, int] = {}
    for key, value in raw.items():
        try:
            mappings[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return mappings


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def default_ports(worker_port: int, *, expose: str = "both") -> tuple[ExposedPort, ...]:
    """The ports to request for a worker listening on ``worker_port``.

    ``both`` asks for the HTTP proxy *and* a direct TCP mapping, so that the
    deployer can prefer the unproxied route and still fall back. ``tcp`` alone
    is the right choice once an orchestrator is known to read the mapping, and
    ``http`` alone is the minimum that works everywhere.
    """
    if expose not in ("http", "tcp", "both"):
        raise SpecError(f"expose must be http, tcp or both, got {expose!r}")
    ports: list[ExposedPort] = []
    if expose in ("http", "both"):
        ports.append(ExposedPort(worker_port, "http"))
    if expose in ("tcp", "both"):
        ports.append(ExposedPort(worker_port, "tcp"))
    return tuple(ports)


def sequence_to_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    """Argparse gives ``None`` for an unused repeatable option."""
    return tuple(values or ())
