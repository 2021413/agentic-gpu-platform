"""The one Modal Volume, and where it is mounted.

The invariant is unchanged from the RunPod image; only the mount point moves:

    immutable image + ephemeral GPU + persistent external volume + runtime config

`PersistentLayout` is imported rather than re-declared so that the directory
names on Modal and on RunPod cannot drift. If the layout ever needs to change,
it changes in `worker/config.py` and both platforms follow.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Final

import modal

from worker.config import PersistentLayout

__all__ = [
    "DATA_ROOT",
    "MODEL_CACHE_VOLUME_NAME",
    "cache_environment",
    "layout",
    "model_cache_volume",
    "volume_mounts",
]

# Not `/runpod-volume`. Nothing on Modal mounts that path, and keeping the
# RunPod name would make the next reader assume a RunPod deployment.
DATA_ROOT: Final = Path("/data")

# Dev and prod share this Volume by default, and that is deliberate. It holds
# 31.2 GB of weights that are byte-identical between the two, are never written
# during serving, and cost an hour of download each time they are not reused.
# Secrets are what must not be shared, and they are not: see `secrets.py`.
MODEL_CACHE_VOLUME_NAME: Final = os.environ.get("MODAL_CACHE_VOLUME", "agentic-gpu-cache")

model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)


def volume_mounts() -> dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]:
    """The `volumes=` argument for every Function and Server in this package."""
    return {str(DATA_ROOT): model_cache_volume}


def layout() -> PersistentLayout:
    """Where weights, caches and the readiness marker live inside a container."""
    return PersistentLayout(DATA_ROOT)


def cache_environment() -> dict[str, str]:
    """Cache redirection for every library that writes gigabytes without asking.

    Baked into the image rather than exported at startup, for the reason the
    Dockerfile already records: `huggingface_hub` and `torch` read these at
    *import* time, so a process started by anything other than our entrypoint —
    `modal shell`, a health probe, an operator's `python -c "import torch"` —
    would otherwise fill the container filesystem instead of the Volume.
    """
    return dict(layout().environment())
