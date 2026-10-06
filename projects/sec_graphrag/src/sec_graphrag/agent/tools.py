"""Tools the agent may call. More arrive with the graph retrievers (M4, M5).

The model never runs a tool itself: it replies with a request ("call search_filings
with these arguments"), our code runs the function, and the result goes back to
the model. The model only sees each tool's name, description and argument schema,
so the docstrings below are effectively prompts.
"""

from langchain_core.tools import tool

from sec_graphrag.companies import FILERS
from sec_graphrag.retrieval.base import Source
from sec_graphrag.retrieval.vector import VectorRetriever

_TICKERS = ", ".join(f.ticker for f in FILERS)


# content_and_artifact: the tool returns (content, artifact). The content (text with
# citation headers) is sent to the model; the artifact (structured sources) stays
# in our code for printing and evals, and is never sent to the model.
@tool(response_format="content_and_artifact", parse_docstring=True)
def search_filings(query: str, ticker: str | None = None) -> tuple[str, list[Source]]:
    """Search the latest annual reports (SEC Form 10-K: Business, Risk Factors, MD&A)
    of 22 US semiconductor companies and return the most relevant excerpts.

    Use this for ANY question about these companies: their business, products,
    customers, suppliers, competitors, risks, strategy or results. Call it again
    with a different query or ticker if the first results are not enough, e.g. once
    per company when comparing companies.

    Args:
        query: What to look for, as a descriptive phrase rather than a single
            keyword (search works by meaning), e.g. "which foundries manufacture
            the company's chips" instead of "foundry".
        ticker: Optional stock ticker to search only one company's report. Omit it
            for questions about several companies or the whole industry.
    """
    if ticker and ticker.upper() not in _TICKERS.split(", "):
        # Returned as content so the model can correct itself, instead of crashing.
        return f"Unknown ticker {ticker!r}. Available tickers: {_TICKERS}.", []
    result = VectorRetriever().retrieve(query, ticker=ticker)
    return result.context, result.sources


# The tool description is built from the docstring; append the ticker list so the
# model knows which companies exist (generated from FILERS, never out of date).
search_filings.description += f"\n\nAvailable tickers: {_TICKERS}."

TOOLS = [search_filings]
