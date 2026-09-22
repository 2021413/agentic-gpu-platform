"""Modal Secrets, and the two that are conspicuously absent.

What a container may need:

    HF_TOKEN      only to fetch a *gated* or rate-limited repository

What it no longer needs:

    VLLM_API_KEY  because on Modal the port is not public
    RUNPOD_API_KEY nothing here provisions anything

On RunPod the inference port was exposed to the internet over direct TCP, so
vLLM's own API key was the only thing between a stranger and a GPU worth several
dollars an hour. A Modal Server is reached exclusively through Modal's proxy,
which rejects an unauthenticated request with 401 before any container sees it.
Running vLLM without a key behind that proxy is therefore not a relaxation: it
removes a secret that had to be distributed, rotated and kept out of logs, and
replaces it with one Modal already manages.

The proxy token is not a Modal Secret and must not be stored as one. It is a
credential the *caller* holds; see `docs/modal.md`.
"""

from __future__ import annotations

import os
from typing import Final

import modal
from modal.exception import NotFoundError

__all__ = ["WORKER_SECRET_NAME", "worker_secrets"]

# Per environment, never shared between dev and prod. `modal secret create`
# takes an `--env` flag for exactly this.
WORKER_SECRET_NAME: Final = os.environ.get("MODAL_WORKER_SECRET", "agentic-gpu-worker")


def worker_secrets() -> list[modal.Secret]:
    """The `secrets=` argument for containers that may reach the hub.

    Optional on purpose. `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` is a public
    repository and needs no credential, so demanding one would fail a deployment
    over a token nobody has a use for. The Secret is attached when it exists and
    skipped, loudly, when it does not.

    Create it only if the model becomes gated, or if anonymous hub rate limits
    start costing download attempts:

        modal secret create agentic-gpu-worker HF_TOKEN=hf_...
    """
    secret = modal.Secret.from_name(WORKER_SECRET_NAME)
    try:
        secret.hydrate()
    except NotFoundError:
        print(
            f"no Modal Secret named {WORKER_SECRET_NAME!r}; continuing without one. "
            f"The model is public, so this is only a problem for a gated repository."
        )
        return []
    return [secret]
