from datetime import date
from itertools import pairwise

from sec_graphrag.companies import Segment
from sec_graphrag.ingest.chunk import (
    CHUNK_TOKENS,
    chunk_filing,
    chunk_id,
    count_tokens,
)
from sec_graphrag.ingest.fetch import FilingDoc


def _doc(items: dict[str, str]) -> FilingDoc:
    return FilingDoc(
        ticker="TEST",
        cik=1,
        company_name="Test Corp",
        segment=Segment.fabless,
        form="10-K",
        accession_no="0000000001-26-000001",
        filing_date=date(2026, 2, 1),
        period_of_report=date(2025, 12, 31),
        items=items,
    )


def _long_text(paragraphs: int) -> str:
    sentence = "We depend on third-party foundries to manufacture our products. "
    return "\n\n".join(f"Paragraph {i}. " + sentence * 20 for i in range(paragraphs))


def test_chunk_id_is_stable_and_readable() -> None:
    assert chunk_id("NVDA", "Item 1A", 7) == "NVDA-1A-0007"
    assert chunk_id("INTC", "Full", 0) == "INTC-Full-0000"


def test_short_section_becomes_one_chunk() -> None:
    chunks = chunk_filing(_doc({"Item 1": "A short business description."}))

    assert len(chunks) == 1
    assert chunks[0].id == "TEST-1-0000"


def test_chunks_respect_token_limit() -> None:
    chunks = chunk_filing(_doc({"Item 1A": _long_text(30)}))

    assert len(chunks) > 1
    assert all(c.n_tokens <= CHUNK_TOKENS for c in chunks)


def test_consecutive_chunks_overlap_by_whole_short_paragraphs() -> None:
    sentence = "We depend on third-party foundries to manufacture our products. "
    short_paragraphs = "\n\n".join(f"Paragraph {i}. {sentence * 3}" for i in range(150))
    chunks = chunk_filing(_doc({"Item 1A": short_paragraphs}))

    # Paragraphs (~40 tokens) fit in the overlap, so each chunk opens with text
    # repeated from the end of the previous one.
    assert len(chunks) > 1
    for prev, nxt in pairwise(chunks):
        assert nxt.text[:60] in prev.text


def test_chunks_never_mix_sections_and_keep_metadata() -> None:
    chunks = chunk_filing(_doc({"Item 1": _long_text(10), "Item 7": _long_text(10)}))

    for item in ("Item 1", "Item 7"):
        indexes = [c.index for c in chunks if c.item == item]
        assert indexes == list(range(len(indexes)))  # numbered per section from 0
    assert {c.accession_no for c in chunks} == {"0000000001-26-000001"}


def test_count_tokens() -> None:
    assert count_tokens("the") == 1
