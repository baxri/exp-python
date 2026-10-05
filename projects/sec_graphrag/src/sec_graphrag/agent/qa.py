"""Answer one question end to end: retrieve context, then generate a cited answer.

Every call is ONE LangSmith trace:

    ask                      (root run: metadata strategy, ticker)
    ├── retrieve             (retriever run: question -> sources)
    └── answer               (LCEL chain: prompt -> llm -> parser)
"""

import time

from langsmith import traceable
from pydantic import BaseModel

from sec_graphrag.agent.answer import get_answer_chain
from sec_graphrag.retrieval.base import RetrievalResult, Retriever, Source, Strategy


class QAResult(BaseModel):
    question: str
    strategy: Strategy
    ticker: str | None
    answer: str
    sources: list[Source]
    cypher: str | None = None
    seconds: float


def get_retriever(strategy: Strategy) -> Retriever:
    """Pick the retriever for a strategy. More strategies arrive in M4-M6."""
    if strategy is Strategy.vector:
        from sec_graphrag.retrieval.vector import VectorRetriever

        return VectorRetriever()
    planned = {Strategy.cypher: "M4", Strategy.hybrid: "M5", Strategy.auto: "M6"}
    raise NotImplementedError(
        f"Strategy {strategy.value!r} is not implemented yet (plan {planned[strategy]})"
    )


def answer_question(
    question: str, strategy: Strategy = Strategy.vector, ticker: str | None = None
) -> QAResult:
    """Retrieve and answer; the whole call is traced as one tree in LangSmith."""
    retriever = get_retriever(strategy)  # fail fast, before any trace or API call
    ticker = ticker.upper() if ticker else None  # "mu" and "MU" are the same company
    start = time.perf_counter()
    answer, retrieval = _ask(
        question,
        retriever,
        ticker,
        # Root-run metadata: lets reports group runs by strategy (M7).
        langsmith_extra={
            "metadata": {"strategy": strategy.value, "ticker": ticker},
            "tags": [f"strategy:{strategy.value}"],
        },
    )
    return QAResult(
        question=question,
        strategy=strategy,
        ticker=ticker,
        answer=answer,
        sources=retrieval.sources,
        cypher=retrieval.cypher,
        seconds=round(time.perf_counter() - start, 2),
    )


@traceable(name="ask")
def _ask(
    question: str, retriever: Retriever, ticker: str | None
) -> tuple[str, RetrievalResult]:
    retrieval = _retrieve(question, retriever, ticker)
    answer = get_answer_chain().invoke(
        {"question": question, "context": retrieval.context},
        config={"run_name": "answer"},
    )
    return answer, retrieval


# run_type="retriever" makes LangSmith show this step as a retrieval step.
@traceable(name="retrieve", run_type="retriever")
def _retrieve(
    question: str, retriever: Retriever, ticker: str | None
) -> RetrievalResult:
    return retriever.retrieve(question, ticker=ticker)
