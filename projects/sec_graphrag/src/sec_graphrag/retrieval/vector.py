"""Vector retriever: find the chunks closest in meaning to the question.

This is the plain-RAG baseline that the graph-based retrievers are compared to.
"""

from functools import cache

from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector

from sec_graphrag.config import get_settings
from sec_graphrag.graph.constraints import VECTOR_INDEX_NAME
from sec_graphrag.ingest.embed import get_embeddings
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
    """LangChain's view of our existing vector index (connects on first use)."""
    settings = get_settings()
    return Neo4jVector.from_existing_index(
        embedding=get_embeddings(),  # the same model that embedded the chunks
        index_name=VECTOR_INDEX_NAME,
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
        # A metadata filter restricts the search to one company's chunks.
        hits = _store().similarity_search_with_score(
            question,
            k=self.k,
            filter={"ticker": ticker.upper()} if ticker else None,
        )
        return to_result(hits)


def to_result(hits: list[tuple[Document, float]]) -> RetrievalResult:
    """Convert LangChain (Document, score) pairs into our RetrievalResult."""
    pieces = [
        (Source(**doc.metadata, score=score), doc.page_content) for doc, score in hits
    ]
    return RetrievalResult(
        context=format_context(pieces),
        sources=[source for source, _ in pieces],
    )
