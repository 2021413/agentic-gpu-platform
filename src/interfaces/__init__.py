"""Delivery mechanisms: HTTP APIs, SSE streaming, CLI entry surfaces.

Nothing here decides *what* the platform does. Every module in this layer
translates a transport-level request into an application use case and renders
the result back onto the transport. Adapters arrive injected: this layer never
imports ``infrastructure``.
"""

from __future__ import annotations
