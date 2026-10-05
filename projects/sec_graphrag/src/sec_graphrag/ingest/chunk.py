"""Split filings into overlapping, token-sized chunks that remember their source.

WHY THIS FILE EXISTS
A 10-K is ~30,000 words. We can't give a whole report (let alone 22 of them) to
the AI for every question: it's too long, too expensive, and the one relevant
paragraph would get lost in the noise. So we cut every report into page-sized
pieces called "chunks" (~1,000 tokens each). Later:
  - step 1.6 turns each chunk into an embedding (a "meaning fingerprint"),
  - at question time we find the few chunks closest in meaning to the question,
  - and only those chunks are given to the AI to write the answer.

WHAT GOES IN AND WHAT COMES OUT
  in:   FilingDoc   one company's report, as {"Item 1": text, "Item 1A": text, ...}
  out:  list[Chunk] ~40 small pieces per company, each labelled with where it came
                    from (company, section, position), so answers can cite sources.
"""

from datetime import date
from functools import cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from sec_graphrag.ingest.fetch import FilingDoc

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Maximum size of one chunk, in tokens (~750 words, roughly one page).
# Big enough to hold a complete thought (e.g. one whole risk factor), small
# enough that its embedding stays about one topic.
CHUNK_TOKENS = 1000

# How much text neighbouring chunks may share at their edges. Without overlap, an
# idea that falls exactly on a cut would be split in two and neither half would
# make sense on its own. (See _splitter() for how overlap really works.)
OVERLAP_TOKENS = 150

# Which tokenizer to count with. AI models read "tokens" (pieces of words, e.g.
# "TSMC" = "T"+"SM"+"C"), and all their limits and prices are in tokens, so we
# measure chunk size in tokens too. "cl100k_base" is the tokenizer used by our
# embedding model (text-embedding-3-small). We must set it explicitly: the
# splitter's default ("gpt2") is an old tokenizer that counts differently.
ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# The data shape of one chunk
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """One piece of one section of one company's 10-K, plus where it came from.

    Each Chunk later becomes a (:Chunk) node in Neo4j, linked to its Filing.
    """

    # Unique, stable ID such as "NVDA-1A-0007" (see chunk_id below).
    id: str
    # Which company: "NVDA". Used to filter searches ("only NVIDIA's chunks").
    ticker: str
    # Which section: "Item 1" (Business), "Item 1A" (Risk Factors),
    # "Item 7" (MD&A), or "Full" for Intel (whole document). Used in citations.
    item: str
    # Position inside its section: 0 = first chunk of that section, 1 = second...
    index: int
    # The actual text of this piece.
    text: str
    # How many tokens the text has (always <= CHUNK_TOKENS). Useful for cost
    # estimates and for checking that chunking worked.
    n_tokens: int
    # The SEC's ID of the filing this chunk came from; links Chunk -> Filing.
    accession_no: str
    # End of the company's fiscal year that the report covers (differs per company).
    period_of_report: date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# @cache: build the splitter once, on first use, and reuse that same object on
# every later call (building it loads the tokenizer, which we don't want to
# repeat ~1,000 times).
@cache
def _splitter() -> RecursiveCharacterTextSplitter:
    """Create LangChain's text splitter, configured to measure in tokens.

    How it decides where to cut ("recursive" = try the gentlest cut first):
      1. split at paragraph breaks ("\\n\\n") and pack whole paragraphs into a chunk
         until the next one would push it over CHUNK_TOKENS;
      2. only if a single paragraph is itself too long, split it at line breaks;
      3. then at spaces between words; 4. as a last resort, inside a word.
    So chunks almost always end at the end of a paragraph, never mid-sentence.

    How overlap really works: the splitter repeats WHOLE paragraphs from the end
    of the previous chunk, as many as fit in OVERLAP_TOKENS. It never cuts a
    paragraph to make overlap. So if the paragraph just before a cut is longer
    than 150 tokens, those two chunks don't overlap at all (about 1 in 3 cuts for
    NVIDIA). That's fine: the cut still falls between complete paragraphs.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
    )


def count_tokens(text: str) -> int:
    """Return how many tokens `text` is, counted exactly as OpenAI counts them."""
    # encode() turns text into a list of token numbers; its length is the count.
    return len(tiktoken.get_encoding(ENCODING).encode(text))


def chunk_id(ticker: str, item: str, index: int) -> str:
    """Build a chunk's ID, e.g. ("NVDA", "Item 1A", 7) -> "NVDA-1A-0007".

    The ID is made only from facts about the chunk, never from randomness, so
    chunking the same filing again always gives the SAME IDs. That matters when
    loading into Neo4j: MERGE on the same ID updates the existing node instead of
    creating a duplicate (our uniqueness constraint on Chunk.id enforces this).
    """
    section = item.removeprefix("Item ").strip()  # "Item 1A" -> "1A"; "Full" stays
    # {index:04d} pads the number to 4 digits (7 -> "0007") so IDs sort correctly:
    # "0010" comes after "0009", whereas "10" would sort before "9".
    return f"{ticker}-{section}-{index:04d}"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def chunk_filing(doc: FilingDoc) -> list[Chunk]:
    """Cut one company's filing into chunks, section by section.

    Each section is split on its own, so a chunk never contains the end of one
    section and the start of the next. Every chunk therefore has exactly one
    `item` label, which keeps citations like [NVDA · Item 1A] accurate.
    """
    chunks: list[Chunk] = []

    # doc.items looks like {"Item 1": "...", "Item 1A": "...", "Item 7": "..."}
    # (or {"Full": "..."} for Intel). Go through the sections one at a time.
    for item, section_text in doc.items.items():
        # Split this section into pieces of at most CHUNK_TOKENS tokens.
        pieces = _splitter().split_text(section_text)

        # enumerate() numbers the pieces 0, 1, 2, ... -> their position (index).
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=chunk_id(doc.ticker, item, index),
                    ticker=doc.ticker,
                    item=item,
                    index=index,
                    text=piece,
                    n_tokens=count_tokens(piece),
                    # Copied from the filing so every chunk knows its source
                    # without having to look anything else up.
                    accession_no=doc.accession_no,
                    period_of_report=doc.period_of_report,
                )
            )

    return chunks
