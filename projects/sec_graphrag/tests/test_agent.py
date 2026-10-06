from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sec_graphrag.agent.graph import searches_for_current_question


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
