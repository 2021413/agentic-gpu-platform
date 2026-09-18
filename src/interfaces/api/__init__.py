"""Public HTTP API (spec sections 3.1 and 20).

The single entry point is ``create_api``; everything else in this package is an
implementation detail of it.
"""

from __future__ import annotations

from interfaces.api.app import create_api

__all__ = ["create_api"]
