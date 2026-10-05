from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sec_graphrag.agent import qa
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


def test_vector_strategy_uses_the_vector_retriever() -> None:
    assert isinstance(qa.get_retriever(Strategy.vector), VectorRetriever)


@pytest.mark.parametrize(
    ("strategy", "milestone"),
    [(Strategy.cypher, "M4"), (Strategy.hybrid, "M5"), (Strategy.auto, "M6")],
)
def test_unbuilt_strategies_say_when_they_arrive(
    strategy: Strategy, milestone: str
) -> None:
    with pytest.raises(NotImplementedError, match=milestone):
        qa.get_retriever(strategy)


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
