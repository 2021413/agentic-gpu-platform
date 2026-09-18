"""Engine and session factory construction.

Deliberately free of any dependency on the composition root: Alembic, the tests
and the future container all need an engine, and none of them should have to
import each other to get one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

__all__ = [
    "DATABASE_URL_ENV",
    "create_database_engine",
    "create_session_factory",
    "database_url_from_env",
    "to_async_url",
]

DATABASE_URL_ENV = "DATABASE_URL"

# Drivers a human (or a test container) may hand us, and which must be rewritten
# because the platform only ever talks to PostgreSQL asynchronously.
_SYNC_PREFIXES = (
    "postgresql+psycopg2://",
    "postgresql+psycopg://",
    "postgresql+pg8000://",
    "postgresql://",
    "postgres://",
)


def to_async_url(url: str) -> str:
    """Force the asyncpg driver.

    A synchronous URL would build a blocking engine, and the first query inside
    the event loop would stall every other request on the same worker.
    """
    for prefix in _SYNC_PREFIXES:
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    return url


def database_url_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Read ``DATABASE_URL``; fail loudly rather than defaulting to localhost."""
    source = environ if environ is not None else os.environ
    url = source.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(f"{DATABASE_URL_ENV} is not set")
    return to_async_url(url)


def create_database_engine(
    url: str,
    *,
    echo: bool = False,
    pool_size: int = 10,
    max_overflow: int = 5,
    pool_timeout: float = 30.0,
    pool_recycle: int = 1800,
) -> AsyncEngine:
    """Build the async engine.

    ``pool_pre_ping`` is on because control-plane replicas outlive the
    connections PostgreSQL or a proxy may drop under them; without it the first
    query after such a drop fails instead of transparently reconnecting.
    ``pool_recycle`` bounds connection age for the same reason.
    """
    return create_async_engine(
        to_async_url(url),
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,
        future=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory used by the unit of work.

    ``expire_on_commit=False`` matters here: the unit of work hands domain
    objects back to callers *after* committing, and an expired attribute would
    trigger a lazy refresh — that is, blocking I/O from inside an object the
    caller believes to be plain data.
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=True,
        class_=AsyncSession,
    )
