from langchain_core.utils.function_calling import convert_to_openai_tool

from sec_graphrag.agent.tools import search_filings
from sec_graphrag.companies import FILERS


def test_schema_the_model_sees() -> None:
    function = convert_to_openai_tool(search_filings)["function"]
    params = function["parameters"]

    assert function["name"] == "search_filings"
    assert params["required"] == ["query"]  # ticker is optional
    assert set(params["properties"]) == {"query", "ticker"}
    # Argument descriptions come from the docstring's Args section.
    assert "search query" in params["properties"]["query"]["description"]
    assert "Args:" not in function["description"]


def test_description_lists_every_filer() -> None:
    assert all(f.ticker in search_filings.description for f in FILERS)


def test_unknown_ticker_returns_a_correctable_message() -> None:
    message = search_filings.invoke(
        {
            "type": "tool_call",
            "id": "call_1",
            "name": "search_filings",
            "args": {"query": "anything", "ticker": "TSLA"},
        }
    )

    assert "Unknown ticker 'TSLA'" in message.content
    assert message.artifact == []
