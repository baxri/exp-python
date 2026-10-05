import pytest

from sec_graphrag.companies import FILERS, Segment, get_filer


def test_tickers_are_unique_and_uppercase() -> None:
    tickers = [f.ticker for f in FILERS]

    assert len(tickers) == len(set(tickers))
    assert all(t == t.upper() for t in tickers)


def test_lookup_is_case_insensitive() -> None:
    assert get_filer("nvda") == get_filer("NVDA")
    assert get_filer("nvda").segment is Segment.fabless


def test_unknown_ticker_raises() -> None:
    with pytest.raises(KeyError, match="NVDIA"):
        get_filer("NVDIA")
