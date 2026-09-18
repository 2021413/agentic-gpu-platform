"""Redis adapters: worker registry, event bus and distributed lock.

Redis is coordination state here, never the source of truth (spec section 14):
what a run did belongs in PostgreSQL, what the platform is doing *right now*
belongs here. Keeping that line means a flushed Redis costs availability and
some in-flight work, never history.
"""

from __future__ import annotations

from infrastructure.redis.client import create_redis_client
from infrastructure.redis.codecs import decode_event, encode_event, event_registry
from infrastructure.redis.event_bus import RedisEventBus, RunEventStream
from infrastructure.redis.keys import DEFAULT_PREFIX, Keyspace
from infrastructure.redis.lock import RedisDistributedLock
from infrastructure.redis.worker_registry import RedisWorkerRegistry

__all__ = [
    "DEFAULT_PREFIX",
    "Keyspace",
    "RedisDistributedLock",
    "RedisEventBus",
    "RedisWorkerRegistry",
    "RunEventStream",
    "create_redis_client",
    "decode_event",
    "encode_event",
    "event_registry",
]
