"""Turn chunk text into embeddings ("meaning fingerprints") and store them in Neo4j.

An embedding is a list of numbers (1,536 for text-embedding-3-small) chosen so
that texts with similar meaning get similar numbers. With an embedding on every
chunk plus a vector index (graph/constraints.py), Neo4j can find the chunks
closest in meaning to a question, even when they share no exact words with it.

Only chunks WITHOUT an embedding are embedded, so re-running is free, and a
chunk whose text changed (its embedding is cleared in graph/load.py) is redone.
"""

from collections.abc import Iterator
from functools import cache

from langchain_openai import OpenAIEmbeddings

from sec_graphrag.config import get_settings
from sec_graphrag.db import run_read, run_write

# Length of the vectors text-embedding-3-small returns. The vector index must be
# created with the same number (tests check that they match).
EMBEDDING_DIMENSIONS = 1536

# Chunks per OpenAI request: 100 chunks are ~90k tokens, well under the API's
# per-request limit, and give 11 progress steps for our ~1,030 chunks.
BATCH_SIZE = 100


@cache
def get_embeddings() -> OpenAIEmbeddings:
    """The one embedding model used for BOTH chunks and questions.

    Vectors from different models live in different "spaces" and can't be
    compared, so everything goes through this single function.
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )


def count_missing() -> int:
    return run_read(
        "MATCH (ch:SEC:Chunk) WHERE ch.embedding IS NULL RETURN count(ch) AS n"
    )[0]["n"]


def embed_missing_chunks() -> Iterator[int]:
    """Embed every chunk that has no embedding yet, one batch at a time.

    Yields the number of chunks embedded after each batch, for progress output.
    """
    while True:
        # 1. Fetch the next batch of chunks that still need an embedding.
        batch = run_read(
            "MATCH (ch:SEC:Chunk) WHERE ch.embedding IS NULL "
            "RETURN ch.id AS id, ch.text AS text LIMIT $limit",
            limit=BATCH_SIZE,
        )
        if not batch:
            return

        # 2. One OpenAI request for the whole batch -> one vector per text,
        #    in the same order as the texts we sent.
        vectors = get_embeddings().embed_documents([row["text"] for row in batch])

        # 3. Write all vectors back in one UNWIND query (one round trip).
        run_write(
            "UNWIND $rows AS row "
            "MATCH (ch:SEC:Chunk {id: row.id}) "
            "SET ch.embedding = row.embedding",
            rows=[
                {"id": row["id"], "embedding": vector}
                for row, vector in zip(batch, vectors, strict=True)
            ],
        )
        yield len(batch)
