"""PostgreSQL persistence adapter (spec section 13).

Layout:

* :mod:`models` — tables, and nothing else;
* :mod:`mappers` — aggregate <-> row translation;
* :mod:`event_codec` — typed (de)serialization of domain events;
* :mod:`repositories` — the persistence ports, implemented;
* :mod:`unit_of_work` — the transaction that keeps state and events in sync;
* :mod:`engine` — engine and session factory construction.
"""

from __future__ import annotations

from infrastructure.database.engine import (
    DATABASE_URL_ENV,
    create_database_engine,
    create_session_factory,
    database_url_from_env,
    to_async_url,
)
from infrastructure.database.models import Base
from infrastructure.database.repositories import (
    SqlAlchemyCandidateRepository,
    SqlAlchemyEventStore,
    SqlAlchemyJobRepository,
    SqlAlchemyPlanRepository,
    SqlAlchemyProjectRepository,
    SqlAlchemyReviewRepository,
    SqlAlchemyRunRepository,
    SqlAlchemyToolResultRepository,
    SqlAlchemyWorkerRepository,
)
from infrastructure.database.unit_of_work import SqlAlchemyUnitOfWork

__all__ = [
    "DATABASE_URL_ENV",
    "Base",
    "SqlAlchemyCandidateRepository",
    "SqlAlchemyEventStore",
    "SqlAlchemyJobRepository",
    "SqlAlchemyPlanRepository",
    "SqlAlchemyProjectRepository",
    "SqlAlchemyReviewRepository",
    "SqlAlchemyRunRepository",
    "SqlAlchemyToolResultRepository",
    "SqlAlchemyUnitOfWork",
    "SqlAlchemyWorkerRepository",
    "create_database_engine",
    "create_session_factory",
    "database_url_from_env",
    "to_async_url",
]
