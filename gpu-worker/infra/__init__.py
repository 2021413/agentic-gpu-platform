"""Deployment adapters. No business logic lives here.

Everything under `infra/` exists to attach this worker to one platform or
another. If a module here starts deciding *what* the worker does rather than
*where* it runs, it belongs in `src/worker/` instead.
"""

__all__: list[str] = []
