"""Vector retriever: find the chunks closest in meaning to the question.

This is the plain-RAG baseline that the graph-based retrievers are compared to.
"""

from functools import cache

from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
from langsmith import traceable

from sec_graphrag.config import get_settings
from sec_graphrag.graph.constraints import VECTOR_INDEX_NAME
from sec_graphrag.ingest.embed import EMBEDDING_DIMENSIONS, get_embeddings
from sec_graphrag.retrieval.base import RetrievalResult, Source, format_context

# How many chunks to retrieve: ~6 x 900 tokens ≈ 5.5k tokens of context.
DEFAULT_K = 6

# Runs once per matched chunk (`node`, `score` come from the vector search) and
# decides what LangChain gets back: the text, plus our source fields as metadata.
RETRIEVAL_QUERY = (
    "RETURN node.text AS text, score, "
    "{chunk_id: node.id, ticker: node.ticker, item: node.item, "
    " period_of_report: toString(node.period_of_report)} AS metadata"
)


@cache
def _store() -> Neo4jVector:
    """LangChain's view of our existing vector index (connects on first use).

    Built directly with the index settings we already know, instead of
    `Neo4jVector.from_existing_index`, which looks them up with extra queries:
    measured ~1s faster to the first result. Connecting to Aura itself (~2.5s) is
    paid once per process either way.
    """
    settings = get_settings()
    return Neo4jVector(
        embedding=get_embeddings(),  # the same model that embedded the chunks
        index_name=VECTOR_INDEX_NAME,
        node_label="Chunk",
        embedding_node_property="embedding",
        text_node_property="text",
        embedding_dimension=EMBEDDING_DIMENSIONS,
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
        retrieval_query=RETRIEVAL_QUERY,
    )


class VectorRetriever:
    def __init__(self, k: int = DEFAULT_K) -> None:
        self.k = k

    def retrieve(self, question: str, *, ticker: str | None = None) -> RetrievalResult:
        # Each step is its own span, so traces show where the time goes.
        if _store.cache_info().currsize == 0:
            _connect()  # first call in this process only
        vector = _embed_question(question)
        hits = _neo4j_search(question, vector, self.k, ticker)
        return to_result(hits)


@traceable(name="neo4j_connect")
def _connect() -> None:
    """One-time setup per process: driver, TLS, login, version check (~2-3s)."""
    _store()


@traceable(name="embed_question", run_type="embedding")
def _embed_question(question: str) -> list[float]:
    return get_embeddings().embed_query(question)


@traceable(name="neo4j_search", process_outputs=lambda hits: {"hits": len(hits)})
def _neo4j_search(
    question: str, vector: list[float], k: int, ticker: str | None
) -> list[tuple[Document, float]]:
    # A metadata filter restricts the search to one company's chunks.
    return _store().similarity_search_with_score_by_vector(
        vector,
        k=k,
        filter={"ticker": ticker.upper()} if ticker else None,
        query=question,  # unused for pure vector search, but the library requires it
    )


def to_result(hits: list[tuple[Document, float]]) -> RetrievalResult:
    """Convert LangChain (Document, score) pairs into our RetrievalResult."""
    pieces = [
        (Source(**doc.metadata, score=score), doc.page_content) for doc, score in hits
    ]
    return RetrievalResult(
        context=format_context(pieces),
        sources=[source for source, _ in pieces],
    )
