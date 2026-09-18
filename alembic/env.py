"""Alembic environment.

Two constraints shape this file:

* it must not import the composition root — migrations run in contexts (a CI
  job, a one-shot container) where no application is wired up, so the database
  URL comes straight from ``DATABASE_URL``;
* the engine is asynchronous, like the application's, so that a migration
  exercising a query behaves exactly as production does.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from infrastructure.database.engine import create_database_engine, database_url_from_env
from infrastructure.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate compares the live database against exactly these tables.
target_metadata = Base.metadata


def _database_url() -> str:
    """Where to migrate, most explicit source first.

    ``-x url=...`` is for ad-hoc targets, ``sqlalchemy.url`` for a caller
    driving Alembic programmatically (the schema tests do), and otherwise the
    environment — never a value committed to this repository.
    """
    override = context.get_x_argument(as_dictionary=True).get("url")
    configured = config.get_main_option("sqlalchemy.url", None)
    return override or configured or database_url_from_env()


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Without these, a widened VARCHAR or a changed server default is
        # silently ignored by autogenerate and the schema drifts.
        compare_type=True,
        compare_server_default=True,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (``alembic upgrade --sql``)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database.

    The pool is kept at a single connection: a migration is one serial stream of
    DDL, and a larger pool only makes a failure harder to reason about.
    """
    engine = create_database_engine(_database_url(), pool_size=1, max_overflow=0)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
