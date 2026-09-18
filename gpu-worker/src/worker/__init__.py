"""Bootstrap for a RunPod GPU worker serving an LLM through vLLM.

The architectural invariant this package exists to enforce:

    immutable image + ephemeral GPU + persistent external volume + runtime config

No model weight is ever baked into a layer, downloaded during the build, or
written to the container filesystem. Destroying and recreating a Pod with the
same volume attached must not re-download 31 GB.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
