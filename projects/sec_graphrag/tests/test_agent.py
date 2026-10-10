from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sec_graphrag.agent import tools
from sec_graphrag.agent.graph import build_agent, searches_for_current_question
from sec_graphrag.retrieval.base import RetrievalResult, Source


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted replies, one per call, instead of calling OpenAI.

    The stock fake model raises on `bind_tools`; the agent needs it, so here it
    simply returns the same model (the script decides about tool calls, not tools).
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        return self


def _fake_model(*replies: AIMessage) -> FakeToolModel:
    messages: Iterator[AIMessage] = iter(replies)
    return FakeToolModel(messages=messages)


def _search_call(call_id: str, ticker: str | None = None) -> dict[str, Any]:
    return {
        "name": "search_filings",
        "args": {"query": "foundry partners", "ticker": ticker},
        "id": call_id,
    }


class FakeRetriever:
    """Stands in for VectorRetriever so the real tool runs without Neo4j."""

    def retrieve(self, question: str, *, ticker: str | None = None) -> RetrievalResult:
        source = Source(
            chunk_id="NVDA-1-0001",
            ticker="NVDA",
            item="Item 1",
            period_of_report=date(2026, 1, 25),
        )
        return RetrievalResult(context="[NVDA · Item 1] We use TSMC.", sources=[source])


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "VectorRetriever", FakeRetriever)


def _tool_result(call_id: str) -> ToolMessage:
    return ToolMessage("excerpts", tool_call_id=call_id)


def test_counts_only_searches_after_the_latest_question() -> None:
    messages = [
        HumanMessage("Who are Micron's competitors?"),
        AIMessage("", tool_calls=[{"name": "search_filings", "args": {}, "id": "a"}]),
        _tool_result("a"),
        AIMessage("Samsung, SK hynix, ..."),
        HumanMessage("Which of them are Chinese?"),  # new question: counting restarts
        AIMessage("", tool_calls=[{"name": "search_filings", "args": {}, "id": "b"}]),
        _tool_result("b"),
    ]

    assert searches_for_current_question(messages) == 1


def test_no_searches_yet() -> None:
    assert searches_for_current_question([HumanMessage("Hi")]) == 0


def test_company_question_loops_through_the_tool() -> None:
    llm = _fake_model(
        AIMessage("", tool_calls=[_search_call("a", ticker="NVDA")]),
        AIMessage("NVDA uses TSMC [NVDA · Item 1]."),
    )

    result = build_agent(llm).invoke(
        {"messages": [HumanMessage("Who makes NVDA chips?")]}
    )

    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    tool_message = result["messages"][2]
    assert "We use TSMC" in tool_message.content  # what the model saw
    assert tool_message.artifact[0].ticker == "NVDA"  # what our code keeps
    assert result["messages"][-1].content == "NVDA uses TSMC [NVDA · Item 1]."


def test_general_question_ends_without_tools() -> None:
    llm = _fake_model(AIMessage("General knowledge (not from the filings): ..."))

    result = build_agent(llm).invoke({"messages": [HumanMessage("What is a foundry?")]})

    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage"]
