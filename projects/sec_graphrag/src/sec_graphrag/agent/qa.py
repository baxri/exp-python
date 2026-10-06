"""Answer one question end to end, by the agent or by a fixed pipeline.

    answer_question(question, strategy, ticker)          ← root run "ask"
    ├── strategy auto (default): the agent decides which searches to run
    └── strategy vector/cypher/hybrid: exactly one retrieval, then the answer chain
        (fixed pipelines give evals a fair, like-for-like comparison)

Every call is ONE LangSmith trace whose inputs are question/strategy/ticker and
whose output is the QAResult, with strategy and ticker as metadata.
"""

import time

from langchain_core.messages import AIMessage, ToolMessage
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel

from sec_graphrag.agent.answer import get_answer_chain
from sec_graphrag.retrieval.base import RetrievalResult, Retriever, Source, Strategy


class Search(BaseModel):
    """One tool call the agent made."""

    tool: str
    args: dict[str, object]


class QAResult(BaseModel):
    question: str
    strategy: Strategy
    ticker: str | None
    answer: str
    sources: list[Source]
    searches: list[Search] = []  # agent only: the tool calls it chose to make
    cypher: str | None = None
    seconds: float


def get_retriever(strategy: Strategy) -> Retriever:
    """The retriever behind a fixed pipeline. More arrive in M4 and M5."""
    if strategy is Strategy.vector:
        from sec_graphrag.retrieval.vector import VectorRetriever

        return VectorRetriever()
    if strategy is Strategy.auto:
        raise ValueError("auto runs the agent, which has no single retriever")
    planned = {Strategy.cypher: "M4", Strategy.hybrid: "M5"}
    raise NotImplementedError(
        f"Strategy {strategy.value!r} is not implemented yet (plan {planned[strategy]})"
    )


@traceable(name="ask")
def answer_question(
    question: str, strategy: Strategy = Strategy.auto, ticker: str | None = None
) -> QAResult:
    """Answer a question; the whole call is one trace in LangSmith."""
    ticker = ticker.upper() if ticker else None  # "mu" and "MU" are the same company
    if run := get_current_run_tree():  # None when tracing is off
        run.add_metadata({"strategy": strategy.value, "ticker": ticker})
        run.add_tags([f"strategy:{strategy.value}"])

    start = time.perf_counter()
    if strategy is Strategy.auto:
        answer, sources, searches, cypher = _run_agent(question, ticker)
    else:
        answer, sources, searches, cypher = _run_pipeline(question, strategy, ticker)
    return QAResult(
        question=question,
        strategy=strategy,
        ticker=ticker,
        answer=answer,
        sources=sources,
        searches=searches,
        cypher=cypher,
        seconds=round(time.perf_counter() - start, 2),
    )


def _run_agent(
    question: str, ticker: str | None
) -> tuple[str, list[Source], list[Search], str | None]:
    from sec_graphrag.agent.graph import get_agent

    content = f"{question}\n\n(Focus on ticker {ticker}.)" if ticker else question
    state = get_agent().invoke(
        {"messages": [("user", content)]}, config={"run_name": "agent"}
    )
    messages = state["messages"]

    searches = [
        Search(tool=call["name"], args=call["args"])
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    # Each tool result carries its sources as an artifact; keep each chunk once.
    sources: dict[str, Source] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.artifact:
            for source in m.artifact:
                sources.setdefault(source.chunk_id, source)
    return messages[-1].text, list(sources.values()), searches, None


def _run_pipeline(
    question: str, strategy: Strategy, ticker: str | None
) -> tuple[str, list[Source], list[Search], str | None]:
    retriever = get_retriever(strategy)  # fails fast for strategies not built yet
    retrieval = _retrieve(question, retriever, ticker)
    answer = get_answer_chain().invoke(
        {"question": question, "context": retrieval.context},
        config={"run_name": "answer"},
    )
    return answer, retrieval.sources, [], retrieval.cypher


# run_type="retriever" makes LangSmith show this as a retrieval step;
# process_inputs keeps the retriever object out of the recorded inputs.
@traceable(
    name="retrieve",
    run_type="retriever",
    process_inputs=lambda inputs: {
        "question": inputs["question"],
        "ticker": inputs["ticker"],
    },
)
def _retrieve(
    question: str, retriever: Retriever, ticker: str | None
) -> RetrievalResult:
    return retriever.retrieve(question, ticker=ticker)
