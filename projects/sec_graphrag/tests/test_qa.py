from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sec_graphrag.agent import graph, qa
from sec_graphrag.agent.answer import build_answer_chain
from sec_graphrag.retrieval.base import RetrievalResult, Source, Strategy
from sec_graphrag.retrieval.vector import VectorRetriever


class FakeRetriever:
    """Returns one fixed chunk and remembers what it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(self, question: str, *, ticker: str | None = None) -> RetrievalResult:
        self.calls.append((question, ticker))
        source = Source(
            chunk_id="NVDA-1-0006",
            ticker="NVDA",
            item="Item 1",
            period_of_report=date(2026, 1, 25),
        )
        return RetrievalResult(context="We use TSMC.", sources=[source])


def _source(chunk_id: str = "NVDA-1-0006") -> Source:
    return Source(
        chunk_id=chunk_id,
        ticker="NVDA",
        item="Item 1",
        period_of_report=date(2026, 1, 25),
    )


def test_vector_strategy_uses_the_vector_retriever() -> None:
    assert isinstance(qa.get_retriever(Strategy.vector), VectorRetriever)


@pytest.mark.parametrize(
    ("strategy", "milestone"),
    [(Strategy.cypher, "M4"), (Strategy.hybrid, "M5")],
)
def test_unbuilt_strategies_say_when_they_arrive(
    strategy: Strategy, milestone: str
) -> None:
    with pytest.raises(NotImplementedError, match=milestone):
        qa.get_retriever(strategy)


def test_auto_has_no_single_retriever() -> None:
    with pytest.raises(ValueError, match="agent"):
        qa.get_retriever(Strategy.auto)


def test_answer_question_wires_retrieval_into_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeRetriever()
    fake_chain = build_answer_chain(
        FakeListChatModel(responses=["TSMC [NVDA · Item 1]"])
    )
    monkeypatch.setattr(qa, "get_retriever", lambda strategy: retriever)
    monkeypatch.setattr(qa, "get_answer_chain", lambda: fake_chain)

    result = qa.answer_question("Who makes the chips?", Strategy.vector, ticker="nvda")

    assert retriever.calls == [("Who makes the chips?", "NVDA")]
    assert result.answer == "TSMC [NVDA · Item 1]"
    assert [s.chunk_id for s in result.sources] == ["NVDA-1-0006"]
    assert result.strategy is Strategy.vector
    assert result.ticker == "NVDA"  # normalized


class FakeAgent:
    """Returns a scripted conversation: one search, its result, then an answer."""

    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def invoke(self, inputs: dict, config: dict | None = None) -> dict:
        self.inputs.append(inputs)
        call = {"name": "search_filings", "args": {"query": "foundries"}, "id": "c1"}
        return {
            "messages": [
                HumanMessage("Who makes the chips?"),
                AIMessage("", tool_calls=[call]),
                ToolMessage(
                    "...", tool_call_id="c1", artifact=[_source(), _source()]
                ),  # same chunk twice: must be kept once
                AIMessage("TSMC makes them [NVDA · Item 1]."),
            ]
        }


def test_auto_runs_the_agent_and_collects_searches_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(graph, "get_agent", lambda: agent)

    result = qa.answer_question("Who makes the chips?", ticker="nvda")

    assert result.strategy is Strategy.auto  # the default
    assert result.answer == "TSMC makes them [NVDA · Item 1]."
    assert result.searches == [
        qa.Search(tool="search_filings", args={"query": "foundries"})
    ]
    assert [s.chunk_id for s in result.sources] == ["NVDA-1-0006"]
    # --ticker becomes a hint in the question, the agent still chooses its searches
    assert "Focus on ticker NVDA" in agent.inputs[0]["messages"][0][1]
