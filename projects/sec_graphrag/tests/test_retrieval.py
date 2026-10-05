from datetime import date

from langchain_core.documents import Document

from sec_graphrag.retrieval.base import Source, format_context
from sec_graphrag.retrieval.vector import to_result


def _source(chunk_id: str = "NVDA-1-0006", item: str = "Item 1") -> Source:
    return Source(
        chunk_id=chunk_id,
        ticker="NVDA",
        item=item,
        period_of_report=date(2026, 1, 25),
    )


def test_source_label() -> None:
    assert _source(item="Item 1A").label == "NVDA · Item 1A"


def test_format_context_puts_a_citable_header_above_each_text() -> None:
    context = format_context(
        [
            (_source(), "We utilize foundries."),
            (_source(item="Item 7"), "Revenue grew."),
        ]
    )

    assert context == (
        "[NVDA · Item 1 | FY ending 2026-01-25]\nWe utilize foundries.\n\n"
        "[NVDA · Item 7 | FY ending 2026-01-25]\nRevenue grew."
    )


def test_to_result_converts_langchain_hits() -> None:
    doc = Document(
        page_content="We utilize foundries, such as TSMC.",
        metadata={
            "chunk_id": "NVDA-1-0006",
            "ticker": "NVDA",
            "item": "Item 1",
            "period_of_report": "2026-01-25",  # Neo4j returns it as a string
        },
    )

    result = to_result([(doc, 0.788)])

    assert result.sources == [_source().model_copy(update={"score": 0.788})]
    assert "TSMC" in result.context
    assert result.cypher is None
