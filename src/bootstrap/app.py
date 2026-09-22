"""The control-plane process.

Assembles the container, hands the HTTP layer its dependencies, and runs the
orchestrator's background loops beside it. This is also where startup and
shutdown are sequenced: resume the runs a previous instance left in flight, and
on the way out let accepted jobs finish rather than killing them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from bootstrap.config import Settings, get_settings
from bootstrap.container import Container, build_container, describe
from bootstrap.logging import configure_logging
from bootstrap.readiness import PlatformReadinessProbe
from interfaces.api.app import create_api
from interfaces.api.dependencies.container import ApiDependencies
from interfaces.worker_api.auth import SharedSecretServiceAuthenticator

__all__ = ["create_app", "main"]

_log = logging.getLogger(__name__)


def _dependencies(container: Container) -> ApiDependencies:
    """Translate the container into exactly what the HTTP layer may reach."""
    return ApiDependencies(
        create_project=container.create_project,
        get_project=container.get_project,
        list_projects=container.list_projects,
        create_run=container.create_run,
        get_run=container.get_run,
        cancel_run=container.cancel_run,
        list_candidates=container.list_candidates,
        list_runs=container.list_runs,
        candidate_patch=container.candidate_patch,
        list_reviews=container.list_reviews,
        approve_run=container.approve_run,
        list_run_events=container.list_run_events,
        list_workers=container.list_workers,
        register_worker=container.register_worker,
        worker_heartbeat=container.heartbeat,
        drain_worker=container.drain_worker,
        deregister_worker=container.deregister_worker,
        event_bus=container.bus,
        readiness=PlatformReadinessProbe(engine=container.engine, redis=container.redis),
        service_authenticator=SharedSecretServiceAuthenticator(
            container.settings.service_token.get_secret_value()
        ),
    )


def create_app(settings: Settings | None = None, *, run_background: bool = True) -> FastAPI:
    """Build the ASGI application.

    ``run_background`` exists so a deployment can serve HTTP without also being
    an orchestrator: the API and the executor are separately scalable, and a
    read-only replica has no business claiming jobs.
    """
    config = settings or get_settings()
    configure_logging(level=config.log_level, fmt=config.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(config)
        app.state.dependencies = _dependencies(container)
        app.state.container = container
        for line in describe(container):
            _log.info("wiring %s", line)

        tasks: list[asyncio.Task[None]] = []
        if run_background:
            # A previous instance may have left runs in flight. Their state is
            # durable; what a restart loses is the intent to act on it.
            resumed = await container.orchestrator.resume_active_runs()
            if resumed:
                _log.info("resumed %d active run(s) after restart", len(resumed))
            tasks.append(asyncio.create_task(container.executor.run_forever(), name="executor"))
            tasks.append(
                asyncio.create_task(container.maintenance.run_forever(), name="maintenance")
            )
        try:
            yield
        finally:
            await container.executor.stop()
            await container.maintenance.stop()
            # Drain rather than cancel: jobs already accepted deserve to finish,
            # and any that cannot are reclaimed through lease expiry anyway.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(container.executor.drain(), timeout=30.0)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await container.aclose()

    return create_api(lifespan=lifespan)


def main() -> None:
    """Console entry point declared in pyproject as ``agentic-api``."""
    config = get_settings()
    uvicorn.run(
        create_app(config),
        host=config.api_host,
        port=config.api_port,
        log_config=None,  # logging is configured by the composition root
    )


if __name__ == "__main__":  # pragma: no cover
    main()
