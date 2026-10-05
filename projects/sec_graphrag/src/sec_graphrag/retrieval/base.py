"""Shared types for all retrievers (vector now; Text2Cypher and hybrid later).

Every retriever takes a question and returns a RetrievalResult of the same shape,
so the answer step, the agent and the evals can use any retriever without
knowing how it finds its text.
"""

from datetime import date
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel


class Strategy(StrEnum):
    """How a question is answered. `auto` lets the router agent decide (M6)."""

    vector = "vector"
    cypher = "cypher"
    hybrid = "hybrid"
    auto = "auto"


class Source(BaseModel):
    """Where one piece of context came from; used for citations."""

    chunk_id: str  # e.g. "NVDA-1A-0007"
    ticker: str
    item: str  # "Item 1A", "Full", ...
    period_of_report: date
    score: float | None = None  # similarity, if the retriever has one

    @property
    def label(self) -> str:
        """Short citation label, e.g. "NVDA · Item 1A"."""
        return f"{self.ticker} · {self.item}"


class RetrievalResult(BaseModel):
    context: str  # the text handed to the LLM, with a source header per piece
    sources: list[Source]
    cypher: str | None = None  # the graph query used (Text2Cypher only)


class Retriever(Protocol):
    """Anything with this `retrieve` method counts as a retriever."""

    def retrieve(self, question: str, *, ticker: str | None = None) -> RetrievalResult:
        """Find text relevant to `question`, optionally only from one company."""
        ...


def format_context(pieces: list[tuple[Source, str]]) -> str:
    """Join retrieved texts, each under a header the LLM can cite.

    [NVDA · Item 1 | FY ending 2026-01-25]
    We utilize foundries, such as ...
    """
    return "\n\n".join(
        f"[{source.label} | FY ending {source.period_of_report}]\n{text}"
        for source, text in pieces
    )
