"""Download 10-K sections from SEC EDGAR and cache them as JSON."""

import re
from datetime import date
from pathlib import Path

from edgar import Company, set_identity
from pydantic import BaseModel

from sec_graphrag.companies import Segment, get_filer
from sec_graphrag.config import get_settings

# The 10-K sections we keep: Business, Risk Factors, MD&A.
ITEMS = ("Item 1", "Item 1A", "Item 7")


class FilingDoc(BaseModel):
    """One company's latest 10-K, reduced to the sections we use."""

    ticker: str
    cik: int
    company_name: str
    segment: Segment
    form: str
    accession_no: str
    filing_date: date
    period_of_report: date
    items: dict[str, str]  # e.g. {"Item 1A": "Item 1A. Risk Factors ..."}


def raw_path(ticker: str) -> Path:
    return get_settings().data_dir / "raw" / f"{ticker.upper()}.json"


def clean_text(text: str) -> str:
    """Remove page artifacts: lone page numbers, 'Table of Contents', blank runs."""
    kept = [
        line
        for line in text.splitlines()
        if not re.fullmatch(r"\s*(\d{1,3}|Table of Contents)\s*", line)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def fetch_filing(ticker: str, *, force: bool = False) -> FilingDoc:
    """Return the cached filing, downloading it first if needed (or if `force`)."""
    path = raw_path(ticker)
    if path.exists() and not force:
        return FilingDoc.model_validate_json(path.read_text())

    doc = _download(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.model_dump_json(indent=2))
    return doc


def _download(ticker: str) -> FilingDoc:
    filer = get_filer(ticker)
    identity = get_settings().edgar_identity
    if not identity:
        raise RuntimeError('EDGAR_IDENTITY is not set (e.g. "Your Name you@email.com")')
    set_identity(identity)

    company = Company(filer.ticker)
    filing = company.get_filings(form="10-K", amendments=False).latest()
    if filing is None:
        raise LookupError(f"No 10-K found for {filer.ticker}")
    tenk = filing.obj()

    return FilingDoc(
        ticker=filer.ticker,
        cik=company.cik,
        company_name=company.name,
        segment=filer.segment,
        form=filing.form,
        accession_no=filing.accession_no,
        filing_date=filing.filing_date,
        period_of_report=filing.period_of_report,
        items={item: clean_text(tenk[item] or "") for item in ITEMS},
    )
