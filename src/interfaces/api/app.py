"""The ASGI application factory.

``create_api`` assembles routers, middleware and error handlers and returns a
FastAPI instance. It **takes** its dependencies; it never constructs them. That
is what keeps this layer free of ``infrastructure``: the composition root
decides what a ``UnitOfWorkFactory`` or an ``EventBus`` really is, and this
module only knows the shape declared in ``ApiDependencies``.

Contract for ``bootstrap``:

.. code-block:: python

    from interfaces.api.app import create_api
    from interfaces.api.dependencies.container import ApiDependencies

    app = create_api(dependencies=ApiDependencies(...))

Or, when the container is only available later (lazy adapters, lifespan):

.. code-block:: python

    app = create_api(dependencies=None)
    app.state.dependencies = ApiDependencies(...)   # before the first request
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from interfaces.api.dependencies.container import ApiDependencies
from interfaces.api.errors import DEFAULT_PROBLEM_BASE_URI, install_error_handlers
from interfaces.api.middleware.correlation import RequestContextMiddleware
from interfaces.api.routes import events, health, projects, runs, workers
from interfaces.worker_api.routes import router as worker_router

__all__ = ["create_api"]

DESCRIPTION = """\
Control plane for a horizontally scalable agentic coding platform.

Every endpoint translates HTTP into an application use case; errors are RFC 9457
problem documents; run progress is streamed as Server-Sent Events.
"""


def create_api(
    *,
    dependencies: ApiDependencies | None = None,
    title: str = "Agentic GPU Platform",
    version: str = "1.0.0",
    problem_base_uri: str = DEFAULT_PROBLEM_BASE_URI,
    include_internal_api: bool = True,
    lifespan: Any = None,
) -> FastAPI:
    """Build the ASGI application.

    ``include_internal_api`` exists because the worker-facing surface may be
    deployed on its own port or its own ingress, reachable from the cluster but
    not from the internet. Serving it from the same process is the local and
    single-node default, not an architectural assumption.

    ``lifespan`` is passed straight through: startup and shutdown (pools,
    background reapers) belong to the composition root, which is the only place
    that knows what needs opening and closing.
    """
    app = FastAPI(
        title=title,
        version=version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    if dependencies is not None:
        app.state.dependencies = dependencies

    # Outermost middleware: the request id must exist before anything can log or
    # fail, including the error handlers that quote it in a problem document.
    app.add_middleware(RequestContextMiddleware)

    install_error_handlers(app, problem_base_uri=problem_base_uri)

    app.include_router(health.router)
    app.include_router(projects.router)
    app.include_router(runs.projects_router)
    app.include_router(runs.router)
    app.include_router(events.router)
    app.include_router(workers.router)
    if include_internal_api:
        app.include_router(worker_router)
    return app
