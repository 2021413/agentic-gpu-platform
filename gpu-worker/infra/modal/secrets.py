"""Modal Secrets, and the one that is conspicuously absent.

What a container needs:

    HF_TOKEN      to fetch a gated or rate-limited repository from the hub

What it no longer needs:

    VLLM_API_KEY  because on Modal the port is not public

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

__all__ = ["WORKER_SECRET_NAME", "worker_secrets"]

# Per environment, never shared between dev and prod. `modal secret create`
# takes an `--env` flag for exactly this.
WORKER_SECRET_NAME: Final = os.environ.get("MODAL_WORKER_SECRET", "agentic-gpu-worker")


def worker_secrets() -> list[modal.Secret]:
    """The `secrets=` argument for containers that may reach the hub.

    Resolution is lazy: a missing Secret fails at `modal deploy` with a message
    naming it, not at 3am inside a container that has already been billed for a
    GPU. Create it with:

        modal secret create agentic-gpu-worker HF_TOKEN=hf_...
    """
    return [modal.Secret.from_name(WORKER_SECRET_NAME)]
