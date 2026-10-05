from pathlib import Path

import pytest

from sec_graphrag.ingest import fetch
from sec_graphrag.ingest.fetch import FetchStatus, FilingDoc, clean_text


def test_removes_lone_page_numbers_and_toc_headers() -> None:
    text = "First paragraph.\n42\nTable of Contents\nSecond paragraph."

    assert clean_text(text) == "First paragraph.\nSecond paragraph."


def test_keeps_numbers_inside_sentences() -> None:
    text = "Revenue grew 42 percent.\n2026 was a record year."

    assert clean_text(text) == text


def test_collapses_blank_line_runs() -> None:
    assert clean_text("A\n\n\n\n\nB") == "A\n\nB"


def test_strips_leading_and_trailing_whitespace() -> None:
    assert clean_text("\n\n  A  \n\n") == "A"


def test_fetch_many_continues_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_fetch_filing(ticker: str, *, force: bool = False) -> FilingDoc:
        if ticker == "BAD":
            raise LookupError("no 10-K")
        return FilingDoc.model_construct(ticker=ticker)  # skip validation: test only

    # Replace the real download and cache location for this test only.
    monkeypatch.setattr(fetch, "fetch_filing", fake_fetch_filing)
    monkeypatch.setattr(fetch, "raw_path", lambda ticker: tmp_path / f"{ticker}.json")

    results = list(fetch.fetch_many(["AAA", "BAD", "CCC"]))

    assert [r.status for r in results] == [
        FetchStatus.downloaded,
        FetchStatus.failed,
        FetchStatus.downloaded,
    ]
    assert "no 10-K" in results[1].error


def test_check_sections_names_every_short_section() -> None:
    long = "word " * fetch.MIN_SECTION_WORDS
    items = {"Item 1": "index only", "Item 1A": long, "Item 7": ""}

    with pytest.raises(
        fetch.SectionError, match=r"Item 1 \(2 words\).*Item 7 \(0 words\)"
    ):
        fetch.check_sections(items)


def test_check_sections_accepts_real_sections() -> None:
    long = "word " * fetch.MIN_SECTION_WORDS
    fetch.check_sections({"Item 1": long, "Item 1A": long, "Item 7": long})
