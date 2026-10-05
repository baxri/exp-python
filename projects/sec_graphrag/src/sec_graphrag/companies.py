"""The companies whose 10-K filings form the dataset."""

from dataclasses import dataclass
from enum import StrEnum


class Segment(StrEnum):
    """Where a company sits in the semiconductor value chain."""

    fabless = "fabless"  # designs chips, outsources manufacturing (e.g. to TSMC)
    idm = "idm"  # integrated device manufacturer: designs and makes its own chips
    equipment = "equipment"  # sells the machines that make chips
    materials = "materials"  # sells chemicals/materials used in fabs
    customer = "customer"  # major buyer of chips, included for demand-side links


@dataclass(frozen=True)
class Filer:
    ticker: str
    segment: Segment


FILERS: list[Filer] = [
    Filer("NVDA", Segment.fabless),
    Filer("AMD", Segment.fabless),
    Filer("QCOM", Segment.fabless),
    Filer("AVGO", Segment.fabless),
    Filer("MRVL", Segment.fabless),
    Filer("MPWR", Segment.fabless),
    Filer("INTC", Segment.idm),
    Filer("TXN", Segment.idm),
    Filer("MU", Segment.idm),
    Filer("ADI", Segment.idm),
    Filer("NXPI", Segment.idm),
    Filer("ON", Segment.idm),
    Filer("MCHP", Segment.idm),
    Filer("SWKS", Segment.idm),
    Filer("QRVO", Segment.idm),
    Filer("WOLF", Segment.idm),
    Filer("AMAT", Segment.equipment),
    Filer("LRCX", Segment.equipment),
    Filer("KLAC", Segment.equipment),
    Filer("TER", Segment.equipment),
    Filer("ENTG", Segment.materials),
    Filer("AAPL", Segment.customer),
]


def get_filer(ticker: str) -> Filer:
    """Look up a filer by ticker (case-insensitive); raises KeyError if unknown."""
    for filer in FILERS:
        if filer.ticker == ticker.upper():
            return filer
    raise KeyError(f"Unknown ticker {ticker!r}")
