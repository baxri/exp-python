from sec_graphrag.ingest.fetch import clean_text


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
