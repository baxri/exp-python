"""Neo4j access: one shared driver per process, plus small read/write helpers."""

import atexit
from functools import lru_cache
from typing import Any, LiteralString

from neo4j import Driver, GraphDatabase, RoutingControl

from sec_graphrag.config import get_settings


@lru_cache
def get_driver() -> Driver:
    """Create the driver (and its connection pool) on first use, then reuse it."""
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )
    # Return pooled connections cleanly when the program exits.
    atexit.register(driver.close)
    return driver


def run_read(query: LiteralString, **params: Any) -> list[dict[str, Any]]:
    """Run a read-only query in a managed (auto-retried) transaction."""
    return _run(query, params, RoutingControl.READ)


def run_write(query: LiteralString, **params: Any) -> list[dict[str, Any]]:
    """Run a write query in a managed (auto-retried) transaction."""
    return _run(query, params, RoutingControl.WRITE)


def run_auto_commit(query: LiteralString, **params: Any) -> list[dict[str, Any]]:
    """Run a query in an auto-commit transaction (no automatic retries).

    Needed for `CALL { ... } IN TRANSACTIONS`, which commits in batches itself and
    therefore cannot run inside a managed transaction.
    """
    with get_driver().session(database=get_settings().neo4j_database) as session:
        return [record.data() for record in session.run(query, params)]


def _run(
    query: LiteralString, params: dict[str, Any], routing: RoutingControl
) -> list[dict[str, Any]]:
    records, _summary, _keys = get_driver().execute_query(
        query,
        params,
        database_=get_settings().neo4j_database,
        routing_=routing,
    )
    return [record.data() for record in records]
