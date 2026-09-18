"""Entry point of the GPU worker process.

Started next to the inference server inside the GPU container. It exits when
told to, and it always tries to deregister on the way out so the pool shrinks
immediately instead of waiting for a heartbeat timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from bootstrap.config import WorkerSettings
from worker_agent.agent import WorkerAgent, WorkerDescription
from worker_agent.client import ControlPlaneClient
from worker_agent.inference import InferenceProbe

__all__ = ["main", "run_worker"]

_log = logging.getLogger(__name__)


def build_agent(settings: WorkerSettings) -> tuple[WorkerAgent, ControlPlaneClient, InferenceProbe]:
    """Assemble the agent from configuration. The only wiring in this process."""
    client = ControlPlaneClient(
        base_url=settings.control_plane_url,
        service_token=settings.service_token.get_secret_value(),
    )
    probe = InferenceProbe(
        base_url=settings.inference_base_url,
        api_key=settings.inference_api_key.get_secret_value(),
    )
    description = WorkerDescription(
        endpoint=settings.worker_endpoint,
        model_id=settings.model_id,
        context_length=settings.model_context_length,
        max_concurrency=settings.worker_concurrency,
        roles=settings.roles,
        gpu_type=settings.gpu_type,
        gpu_count=settings.gpu_count,
        tensor_parallel_size=settings.tensor_parallel_size,
        worker_id=settings.worker_id,
    )
    agent = WorkerAgent(
        client=client,
        probe=probe,
        description=description,
        heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
        drain_timeout_seconds=settings.worker_drain_timeout_seconds,
    )
    return agent, client, probe


async def run_worker(settings: WorkerSettings | None = None) -> None:
    config = settings or WorkerSettings()
    logging.basicConfig(level=config.log_level)
    agent, client, probe = build_agent(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            # A container stop must drain, not kill: jobs in hand deserve to
            # finish, and the pool should shrink deliberately.
            loop.add_signal_handler(sig, agent.stop)

    try:
        await agent.run()
    finally:
        await client.aclose()
        await probe.aclose()


def main() -> None:
    """Console entry point declared in pyproject as ``agentic-worker``."""
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        _log.info("interrupted")


if __name__ == "__main__":  # pragma: no cover
    main()
