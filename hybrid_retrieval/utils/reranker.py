"""
Polished version of the Cohere cross-encoder rerank built in 5-rerank.py.
6-evaluate.py imports from here so the eval script can focus on the metric.
"""

import os
import time
import cohere
import pandas as pd
from cohere.errors import TooManyRequestsError
from dotenv import load_dotenv
from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-fast"

# Trial keys allow 10 calls/minute, so space calls at least 6 seconds apart.
# Set to 0 with a production key.
MIN_SECONDS_BETWEEN_CALLS = 6.5
_last_call_at = 0.0


def _wait_for_rate_limit() -> None:
    """Sleep just long enough to keep calls MIN_SECONDS_BETWEEN_CALLS apart."""
    global _last_call_at
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def rerank_with_cohere(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    max_retries: int = 3,
) -> list[tuple[str, float]]:
    documents = [corpus_by_id.loc[d, "text"] for d in candidate_ids]
    for attempt in range(max_retries + 1):
        _wait_for_rate_limit()
        try:
            response = co.rerank(
                model=RERANK_MODEL, query=query, documents=documents, top_n=k
            )
            break
        except TooManyRequestsError:
            if attempt == max_retries:
                raise
            # Still rate-limited (e.g. calls from an earlier run); let the
            # one-minute window reset before trying again.
            time.sleep(60)
    return [(candidate_ids[r.index], r.relevance_score) for r in response.results]


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank_with_cohere(query, candidate_ids, corpus_by_id, k=k)
