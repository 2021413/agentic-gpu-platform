"""Connection factory for the Redis adapters.

The adapters take a client, they never create one: a process needs exactly one
connection pool, and who owns it is a composition-root decision. This helper
only fixes the two settings every adapter here depends on — responses decoded
to ``str`` (the codecs deal in strings) and periodic health checks, without
which a pooled connection silently rots behind a NAT or a load balancer.

``decode_responses`` is invisible to the type checker: the driver declares
every reply as ``bytes | str`` regardless. The adapters therefore narrow what
they read through ``codecs.as_text``, which holds whether or not the client
came from here.
"""

from __future__ import annotations

from redis.asyncio import Redis

__all__ = ["DEFAULT_HEALTH_CHECK_SECONDS", "create_redis_client"]

DEFAULT_HEALTH_CHECK_SECONDS = 30


def create_redis_client(
    url: str,
    *,
    health_check_interval: int = DEFAULT_HEALTH_CHECK_SECONDS,
    max_connections: int | None = None,
) -> Redis:
    """Build a client the adapters can use. The caller closes it."""
    return Redis.from_url(
        url,
        decode_responses=True,
        health_check_interval=health_check_interval,
        max_connections=max_connections,
    )
