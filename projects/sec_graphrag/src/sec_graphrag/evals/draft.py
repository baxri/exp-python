"""Draft eval question candidates with an LLM, grounded in filing chunks (step 2.2).

The LLM never invents questions from memory: each draft starts from a *context*,
a few real chunks picked for one category, and the question and its reference
answer must come from those chunks only. A human then reviews every candidate
(step 2.3) before it enters `dataset.jsonl`.

How chunks are picked per category (by keyword, from the filing text, never from
the knowledge graph, so the dataset does not favour graph retrieval):

- lookup:      one chunk.
- multi_hop:   two chunks from different companies that mention the same entity
               (e.g. both name TSMC), so the answer needs both filings.
- aggregation: up to 3 chunks from EVERY company that mentions the entity, so
               the reference answer can list all of them. One chunk per company
               was not enough: the trial missed Qualcomm's "Apple is 10%+ of
               revenue" because a different Qualcomm chunk mentioned Apple more.
- thematic:    risk-factor chunks from a few companies on the same theme.
"""

import random
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, Field

from sec_graphrag.companies import FILERS
from sec_graphrag.evals.models import Category, EvalExample
from sec_graphrag.ingest.chunk import Chunk, chunk_filing
from sec_graphrag.ingest.fetch import fetch_filing


@dataclass(frozen=True)
class Topic:
    name: str
    pattern: str  # case-sensitive regex that finds the topic in chunk text


# Named companies/programmes that 4+ filings mention (counted on 2026-10-10).
# A mention is not yet a relationship (Intel may be a customer OR a competitor);
# the drafting prompt asks the LLM to read what the link actually is.
ENTITIES = [
    Topic("TSMC", r"\bTSMC\b|Taiwan Semiconductor Manufacturing"),
    Topic("Samsung", r"\bSamsung\b"),
    Topic("Apple", r"\bApple\b"),
    Topic("Intel", r"\bIntel\b"),
    Topic("Huawei", r"\bHuawei\b"),
    Topic("SK hynix", r"SK [Hh]ynix"),
    Topic("Microsoft", r"\bMicrosoft\b"),
    Topic("Amazon", r"\bAmazon\b"),
    Topic("Google", r"\bGoogle\b|\bAlphabet\b"),
    Topic("CHIPS Act", r"CHIPS Act"),
    Topic("hyperscalers", r"[Hh]yperscal"),
]

# Broad themes that most filings discuss, mainly in Risk Factors.
THEMES = [
    Topic("export controls", r"[Ee]xport control"),
    Topic("tariffs", r"[Tt]ariff"),
    Topic("cybersecurity", r"[Cc]ybersecurity|[Cc]yber-?attack"),
    Topic("climate change", r"[Cc]limate change"),
    Topic("AI demand", r"[Aa]rtificial intelligence|\bAI\b"),
    Topic("talent", r"[Tt]alent|skilled (employees|personnel)"),
    Topic("excess inventory", r"[Ee]xcess inventor"),
    Topic("Taiwan and geopolitics", r"\bTaiwan\b"),
]


@dataclass(frozen=True)
class DraftContext:
    """The chunks one batch of candidate questions is drafted from."""

    category: Category
    chunks: tuple[Chunk, ...]
    topic: str | None = None  # None for lookup


def load_chunks() -> list[Chunk]:
    """All chunks of all cached filings, chunked exactly as for the vector index,
    so chunk IDs in the dataset match the IDs stored in Neo4j."""
    return [chunk for f in FILERS for chunk in chunk_filing(fetch_filing(f.ticker))]


def chunks_per_company(
    chunks: list[Chunk], topic: Topic, limit: int = 1
) -> dict[str, list[Chunk]]:
    """For each company that mentions the topic, its `limit` chunks with the most
    mentions (most first)."""
    found: dict[str, list[tuple[int, Chunk]]] = {}
    for chunk in chunks:
        if hits := len(re.findall(topic.pattern, chunk.text)):
            found.setdefault(chunk.ticker, []).append((hits, chunk))
    return {
        ticker: [c for _, c in sorted(pairs, key=lambda p: -p[0])[:limit]]
        for ticker, pairs in found.items()
    }


def lookup_contexts(
    chunks: list[Chunk], n: int, rng: random.Random
) -> list[DraftContext]:
    # Skip short chunks (section tails, headings): too little to ask about.
    pool = [c for c in chunks if c.n_tokens >= 300]
    return [DraftContext(Category.lookup, (c,)) for c in rng.sample(pool, n)]


def multi_hop_contexts(
    chunks: list[Chunk], n: int, rng: random.Random
) -> list[DraftContext]:
    contexts: list[DraftContext] = []
    seen: set[tuple[str, ...]] = set()
    # Go round the entities so the questions spread over all of them.
    for i in range(n):
        topic = ENTITIES[i % len(ENTITIES)]
        by_company = chunks_per_company(chunks, topic)
        # Avoid asking about the same pair of companies twice for one entity.
        for _ in range(20):
            pair = sorted(rng.sample(sorted(by_company), 2))
            if (topic.name, *pair) not in seen:
                break
        seen.add((topic.name, *pair))
        contexts.append(
            DraftContext(
                Category.multi_hop, tuple(by_company[t][0] for t in pair), topic.name
            )
        )
    return contexts


def aggregation_contexts(chunks: list[Chunk]) -> list[DraftContext]:
    # One context per entity, with every company that mentions it: an aggregation
    # reference answer is only correct if it is complete.
    return [
        DraftContext(
            Category.aggregation,
            tuple(c for t in sorted(by_company) for c in by_company[t]),
            topic.name,
        )
        for topic in ENTITIES
        if (by_company := chunks_per_company(chunks, topic, limit=3))
    ]


def thematic_contexts(
    chunks: list[Chunk], per_theme: int, rng: random.Random
) -> list[DraftContext]:
    risk_chunks = [c for c in chunks if c.item in ("Item 1A", "Full")]
    contexts: list[DraftContext] = []
    for topic in THEMES:
        by_company = chunks_per_company(risk_chunks, topic)
        picked = rng.sample(sorted(by_company), min(per_theme, len(by_company)))
        contexts.append(
            DraftContext(
                Category.thematic, tuple(by_company[t][0] for t in picked), topic.name
            )
        )
    return contexts


# ---------------------------------------------------------------------------
# Drafting with the LLM (step 2.2b)
# ---------------------------------------------------------------------------


# The shape the LLM must fill in. With `with_structured_output`, these classes
# become a JSON schema sent to the model, and its reply is parsed and validated
# back into these objects: no free text to parse ourselves. The Field
# descriptions are part of the prompt the model sees.
class DraftQuestion(BaseModel):
    question: str = Field(
        description="A question an analyst might ask, answerable from the excerpts."
    )
    reference_answer: str = Field(
        description="A short, complete, correct answer using only the excerpts."
    )
    expected_entities: list[str] = Field(
        description="1 to 5 key names or numbers that any correct answer must "
        'mention, e.g. ["TSMC", "50%"]: no dates, no phrases. Use the shortest '
        'form likely to appear in an answer ("Qorvo", not "Qorvo, Inc.").'
    )
    source_chunks: list[str] = Field(
        description="IDs of the excerpts the answer comes from, copied exactly "
        'from their headers, e.g. ["NVDA-1-0006"].'
    )


class DraftBatch(BaseModel):
    questions: list[DraftQuestion] = Field(
        description="The drafted questions; empty if the excerpts support none."
    )


# What a good question looks like for each category.
CATEGORY_GUIDE = {
    Category.lookup: (
        "Ask for ONE specific fact stated in the excerpt (a name, number, product, "
        "date or policy) about the company that filed it."
    ),
    Category.multi_hop: (
        "Ask a question that needs BOTH companies' excerpts to answer, about how "
        "each relates to {topic} (e.g. which supplier both use, or how their links "
        "to {topic} differ). The answer must use facts from both excerpts."
    ),
    Category.aggregation: (
        "Ask which (or how many) companies have ONE specific kind of link to "
        "{topic}, e.g. name it as a customer, supplier or competitor. Do not list "
        "the companies in the question: a user would not know them. The excerpts "
        "are ALL the passages that mention {topic}, so check every company's "
        "excerpts and list every company that qualifies."
    ),
    Category.thematic: (
        "Ask how these companies discuss {topic}: what they share and how they "
        "differ. The answer summarises each company's point with its specifics."
    ),
}

SYSTEM = """You write evaluation questions for a question-answering system over the annual \
reports (SEC Form 10-K) of semiconductor companies. Each question will be asked \
to the system later, and your reference answer is used to grade its answer.

Rules:
- Use ONLY the excerpts below. Never add facts from your own knowledge, even if \
you are sure they are true.
- Name companies in the question; never write "this company", "the excerpt(s)" \
or "these companies": the system will not see the excerpts. For a group, say \
"the semiconductor companies" or "which companies".
- When a company is mentioned, read what the link actually is (customer, \
supplier, competitor, partner...). A mention alone is not a relationship.
- If a company's own report mentions itself, it is not a link to itself.
- Prefer facts that would still be true for the whole fiscal year, and mention \
the fiscal year when a number depends on it.
- Keep it short: a question of at most 30 words, as a user would type it; a \
reference answer of at most 80 words (lookup: one or two sentences).
- Before answering, check that the answer agrees with itself and with every \
excerpt. If you are unsure, drop the question.
- Write fewer questions, or none, rather than a vague or weakly supported one.

Category: {category}
{guide}

Write up to {n} different questions."""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Excerpts:\n\n{excerpts}")]
)


def format_excerpts(chunks: tuple[Chunk, ...]) -> str:
    """Each excerpt under a header with its exact chunk ID, which the model
    must copy into `source_chunks`."""
    return "\n\n".join(
        f"[{c.id} | {c.ticker} · {c.item} | FY ending {c.period_of_report}]\n{c.text}"
        for c in chunks
    )


@traceable(name="draft_questions")
def draft_questions(
    context: DraftContext, llm: BaseChatModel, n: int = 1
) -> list[EvalExample]:
    """Ask the LLM for up to `n` candidates from one context.

    Candidates citing chunk IDs that are not in the context are dropped: the
    model made them up, so we cannot trust that answer to be grounded.
    """
    chain = prompt | llm.with_structured_output(DraftBatch)
    batch = chain.invoke(
        {
            "category": context.category.value,
            "guide": CATEGORY_GUIDE[context.category].format(topic=context.topic),
            "n": n,
            "excerpts": format_excerpts(context.chunks),
        }
    )
    return to_examples(batch, context)


def to_examples(batch: DraftBatch, context: DraftContext) -> list[EvalExample]:
    """Turn the LLM's drafts into dataset examples, keeping only grounded ones."""
    allowed = {c.id for c in context.chunks}
    return [
        EvalExample(category=context.category, **draft.model_dump())
        for draft in batch.questions
        if draft.source_chunks and set(draft.source_chunks) <= allowed
    ]


# ---------------------------------------------------------------------------
# Drafting everything (step 2.2c)
# ---------------------------------------------------------------------------

CANDIDATES_PATH = Path(__file__).parent / "candidates.jsonl"

# About twice the dataset target per category (15/25/10/10), since review rejects
# some. Each job is (context, how many questions to ask for).
LOOKUPS = 30
MULTI_HOPS = 50
QUESTIONS_PER_AGGREGATION = 2  # 11 entities -> up to 22
QUESTIONS_PER_THEME = 2  # 8 themes -> up to 16
COMPANIES_PER_THEME = 3


def all_jobs(chunks: list[Chunk], seed: int = 42) -> list[tuple[DraftContext, int]]:
    """Every context to draft from. A fixed seed picks the same chunks each run."""
    rng = random.Random(seed)
    return [
        *((c, 1) for c in lookup_contexts(chunks, LOOKUPS, rng)),
        *((c, 1) for c in multi_hop_contexts(chunks, MULTI_HOPS, rng)),
        *((c, QUESTIONS_PER_AGGREGATION) for c in aggregation_contexts(chunks)),
        *(
            (c, QUESTIONS_PER_THEME)
            for c in thematic_contexts(chunks, COMPANIES_PER_THEME, rng)
        ),
    ]


def draft_all(
    jobs: list[tuple[DraftContext, int]], llm: BaseChatModel, workers: int = 8
) -> Iterator[list[EvalExample] | Exception]:
    """Draft every job, `workers` LLM calls at a time, yielding results as they
    finish. A failed call yields its exception instead of stopping the run."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(draft_questions, ctx, llm, n) for ctx, n in jobs]
        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as error:  # noqa: BLE001 - reported, run continues
                yield error
