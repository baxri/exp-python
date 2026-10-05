import re

from sec_graphrag.graph.constraints import CONSTRAINTS, VECTOR_INDEX
from sec_graphrag.ingest.embed import EMBEDDING_DIMENSIONS

PATTERN = re.compile(
    r"CREATE CONSTRAINT (?P<name>\w+) IF NOT EXISTS "
    r"FOR \(n:(?P<label>\w+)\) REQUIRE n\.(?P<prop>\w+) IS UNIQUE"
)


def _parsed() -> list[re.Match[str]]:
    matches = [PATTERN.fullmatch(statement) for statement in CONSTRAINTS]
    assert all(matches), "every statement must match the expected idempotent form"
    return [m for m in matches if m]


def test_every_statement_is_idempotent_and_well_formed() -> None:
    _parsed()


def test_names_are_prefixed_and_unique() -> None:
    names = [m["name"] for m in _parsed()]

    assert all(name.startswith("sec_") for name in names)
    assert len(names) == len(set(names))


def test_every_schema_label_has_a_key() -> None:
    keys = {(m["label"], m["prop"]) for m in _parsed()}

    assert keys == {
        ("Company", "name"),
        ("Company", "cik"),
        ("Filing", "accession_no"),
        ("Chunk", "id"),
        ("Person", "name"),
        ("RiskFactor", "name"),
        ("Product", "name"),
        ("Technology", "name"),
        ("Location", "name"),
    }


def test_vector_index_matches_embedding_model() -> None:
    match = re.search(r"`vector.dimensions`: (\d+)", VECTOR_INDEX)

    assert "IF NOT EXISTS" in VECTOR_INDEX
    assert match is not None
    assert int(match[1]) == EMBEDDING_DIMENSIONS
