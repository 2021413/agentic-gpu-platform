"""Modal adaptation of the GPU worker.

The worker itself does not know Modal exists. This package supplies the four
things Modal needs — an image, a Volume, Secrets and a Server class — and then
hands control to `worker.config`, `worker.model_state` and `worker.readiness`,
which are the same modules the RunPod image runs.

Two invocation styles both work, because every import below is absolute and
this directory is a real package:

    modal serve  infra/modal/app.py     # hot reload, development
    modal deploy infra/modal/app.py     # a persistent deployment

Both must be run from the `gpu-worker/` directory: Modal puts the current
working directory on `sys.path`, which is what makes `infra.modal.*` and
`worker.*` importable at deploy time.
"""

__all__: list[str] = []
