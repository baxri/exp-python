"""Write the structural graph to Neo4j: Company -> Filing -> Chunk.

"Structural" = everything we know for certain without AI: which company filed
which report, and which chunks the report was cut into. AI-extracted entities
(TSMC, Taiwan, risks...) are added in M3 and attach to these Chunk nodes.

    (:SEC:Company)-[:FILED]->(:SEC:Filing)-[:HAS_CHUNK]->(:SEC:Chunk)

Every write uses MERGE on a key protected by a uniqueness constraint (step 0.6),
so loading the same filing again updates it instead of duplicating it.
"""

from sec_graphrag.db import run_write
from sec_graphrag.ingest.chunk import Chunk
from sec_graphrag.ingest.fetch import FilingDoc


def load_filing(doc: FilingDoc, chunks: list[Chunk]) -> None:
    """Create or update one company, its filing and all of its chunks."""
    _merge_company_and_filing(doc)
    _merge_chunks(doc.accession_no, chunks)
    _delete_stale_chunks(doc.accession_no, [c.id for c in chunks])


def _merge_company_and_filing(doc: FilingDoc) -> None:
    run_write(
        # Filers are identified by CIK (never changes), not by name or ticker.
        "MERGE (c:SEC:Company {cik: $cik}) "
        "SET c.name = $name, c.ticker = $ticker, c.segment = $segment, "
        "    c.is_filer = true "
        "MERGE (f:SEC:Filing {accession_no: $accession_no}) "
        "SET f.form = $form, f.filing_date = $filing_date, "
        "    f.period_of_report = $period_of_report "
        "MERGE (c)-[:FILED]->(f)",
        cik=doc.cik,
        name=doc.company_name,
        ticker=doc.ticker,
        segment=doc.segment.value,
        accession_no=doc.accession_no,
        form=doc.form,
        # Python `date` objects become native Neo4j dates automatically.
        filing_date=doc.filing_date,
        period_of_report=doc.period_of_report,
    )


def _merge_chunks(accession_no: str, chunks: list[Chunk]) -> None:
    """Write all chunks of one filing in ONE query (one network round trip)."""
    run_write(
        "MATCH (f:SEC:Filing {accession_no: $accession_no}) "
        # UNWIND turns the list parameter into one row per chunk, server-side.
        "UNWIND $rows AS row "
        "MERGE (ch:SEC:Chunk {id: row.id}) "
        # If the text changed, its old embedding (added in step 1.6) no longer
        # matches it, so clear it; it gets re-embedded on the next build.
        "SET ch.embedding = CASE WHEN ch.text = row.text "
        "                        THEN ch.embedding ELSE null END "
        "SET ch.text = row.text, ch.ticker = row.ticker, ch.item = row.item, "
        "    ch.index = row.index, ch.n_tokens = row.n_tokens, "
        "    ch.period_of_report = row.period_of_report "
        "MERGE (f)-[:HAS_CHUNK]->(ch)",
        accession_no=accession_no,
        rows=[chunk.model_dump() for chunk in chunks],
    )


def _delete_stale_chunks(accession_no: str, current_ids: list[str]) -> None:
    """Remove this filing's chunks that are no longer produced by chunking.

    MERGE only adds and updates; it never deletes. Without this step, changing
    the chunk size would leave old, overlapping chunks behind in the database.
    """
    run_write(
        "MATCH (:SEC:Filing {accession_no: $accession_no})-[:HAS_CHUNK]->(ch) "
        "WHERE NOT ch.id IN $current_ids "
        "DETACH DELETE ch",
        accession_no=accession_no,
        current_ids=current_ids,
    )
