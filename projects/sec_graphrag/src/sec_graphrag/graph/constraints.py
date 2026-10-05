"""Uniqueness constraints for the project's graph, and wiping project data."""

from typing import LiteralString

from sec_graphrag.db import run_auto_commit, run_read, run_write

# One statement per constraint. Each also creates an index on the property, which
# makes MERGE and lookups by key fast. IF NOT EXISTS makes the setup idempotent.
CONSTRAINTS: list[LiteralString] = [
    # Structural nodes
    (
        "CREATE CONSTRAINT sec_company_name IF NOT EXISTS "
        "FOR (n:Company) REQUIRE n.name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_company_cik IF NOT EXISTS "
        "FOR (n:Company) REQUIRE n.cik IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_filing_accession IF NOT EXISTS "
        "FOR (n:Filing) REQUIRE n.accession_no IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_chunk_id IF NOT EXISTS "
        "FOR (n:Chunk) REQUIRE n.id IS UNIQUE"
    ),
    # Extracted entities (filled in M3)
    (
        "CREATE CONSTRAINT sec_person_name IF NOT EXISTS "
        "FOR (n:Person) REQUIRE n.name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_risk_name IF NOT EXISTS "
        "FOR (n:RiskFactor) REQUIRE n.name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_product_name IF NOT EXISTS "
        "FOR (n:Product) REQUIRE n.name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_technology_name IF NOT EXISTS "
        "FOR (n:Technology) REQUIRE n.name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT sec_location_name IF NOT EXISTS "
        "FOR (n:Location) REQUIRE n.name IS UNIQUE"
    ),
]


def create_constraints() -> None:
    """Create all constraints; safe to run any number of times."""
    for statement in CONSTRAINTS:
        run_write(statement)


def list_constraints() -> list[dict[str, str]]:
    """Return the project's constraints as {name, label, property} rows."""
    return run_read(
        "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
        "WHERE name STARTS WITH 'sec_' "
        "RETURN name, labelsOrTypes[0] AS label, properties[0] AS property "
        "ORDER BY name"
    )


def count_project_nodes() -> int:
    return run_read("MATCH (n:SEC) RETURN count(n) AS count")[0]["count"]


def wipe_project_data() -> int:
    """Delete every :SEC node and its relationships, in batches. Returns the count."""
    count = count_project_nodes()
    # Batched so a large graph never has to fit in one transaction's memory.
    run_auto_commit(
        "MATCH (n:SEC) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
    )
    return count
