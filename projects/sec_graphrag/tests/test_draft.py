import random
import re
from datetime import date

import pytest

from sec_graphrag.evals.draft import (
    ENTITIES,
    THEMES,
    DraftBatch,
    DraftContext,
    DraftQuestion,
    Topic,
    aggregation_contexts,
    chunks_per_company,
    multi_hop_contexts,
    to_examples,
)
from sec_graphrag.evals.models import Category
from sec_graphrag.ingest.chunk import Chunk

TSMC = Topic("TSMC", r"\bTSMC\b")


def _chunk(ticker: str, index: int, text: str) -> Chunk:
    return Chunk(
        id=f"{ticker}-1-{index:04d}",
        ticker=ticker,
        item="Item 1",
        index=index,
        text=text,
        n_tokens=400,
        accession_no="0000000000-26-000001",
        period_of_report=date(2026, 1, 1),
    )


CHUNKS = [
    _chunk("NVDA", 0, "We use TSMC."),
    _chunk("NVDA", 1, "TSMC makes wafers. TSMC also packages them."),  # most mentions
    _chunk("AMD", 0, "We rely on TSMC for 7nm."),
    _chunk("MU", 0, "We run our own fabs."),  # never mentions TSMC
]


def test_chunks_with_most_mentions_come_first() -> None:
    found = chunks_per_company(CHUNKS, TSMC, limit=2)

    assert set(found) == {"NVDA", "AMD"}  # MU never mentions TSMC
    assert [c.id for c in found["NVDA"]] == ["NVDA-1-0001", "NVDA-1-0000"]
    assert [c.id for c in chunks_per_company(CHUNKS, TSMC)["NVDA"]] == ["NVDA-1-0001"]


def test_multi_hop_pairs_two_different_companies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sec_graphrag.evals.draft.ENTITIES", [TSMC])

    [context] = multi_hop_contexts(CHUNKS, 1, random.Random(0))

    assert context.category is Category.multi_hop
    assert context.topic == "TSMC"
    assert {c.ticker for c in context.chunks} == {"NVDA", "AMD"}


def test_aggregation_includes_every_company_that_mentions_the_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sec_graphrag.evals.draft.ENTITIES", [TSMC])

    [context] = aggregation_contexts(CHUNKS)

    # Every company, sorted, with up to 3 chunks each (NVDA has 2).
    assert [c.id for c in context.chunks] == [
        "AMD-1-0000",
        "NVDA-1-0001",
        "NVDA-1-0000",
    ]


def test_topic_patterns_are_valid_regexes() -> None:
    for topic in [*ENTITIES, *THEMES]:
        re.compile(topic.pattern)


def _draft(*chunk_ids: str) -> DraftQuestion:
    return DraftQuestion(
        question="Which foundry does NVIDIA use?",
        reference_answer="TSMC.",
        expected_entities=["TSMC"],
        source_chunks=list(chunk_ids),
    )


def test_drafts_citing_unknown_chunks_are_dropped() -> None:
    context = DraftContext(Category.lookup, (CHUNKS[1],))
    batch = DraftBatch(
        questions=[
            _draft("NVDA-1-0001"),  # the chunk we gave: kept
            _draft("NVDA-1-0099"),  # invented ID: dropped
            _draft(),  # no source at all: dropped
        ]
    )

    [example] = to_examples(batch, context)

    assert example.category is Category.lookup
    assert example.source_chunks == ["NVDA-1-0001"]
