"""The tool-calling agent, built by hand as a LangGraph StateGraph.

    START ──▶ agent ──(tool calls?)──yes──▶ tools ──┐
                ▲            │                       │
                │            no                      │
                │            ▼                       │
                │           END                      │
                └────────────────────────────────────┘

- `agent`: the LLM (with our tools bound) reads the conversation and either
  answers or asks for tool calls.
- `tools`: runs every requested tool and appends the results as ToolMessages.
- The loop repeats until the LLM answers without asking for a tool.
"""

from functools import cache
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from sec_graphrag.agent.tools import TOOLS
from sec_graphrag.config import get_settings

# Standing instructions, sent before the conversation on every LLM call.
SYSTEM_PROMPT = """\
You are an analyst assistant for the latest annual reports (SEC Form 10-K) of 22 US \
semiconductor companies. The `search_filings` tool searches those reports.

What the data contains: only Item 1 (Business), Item 1A (Risk Factors) and Item 7 \
(MD&A) of each report (Intel: its full report). It does NOT contain stock prices, \
financial statements, exhibits or anything published after the filing.

Grounding rule (most important):
- For ANY question about these companies or the semiconductor industry's companies, \
including their business, products, customers, suppliers, manufacturing, competitors, \
risks, strategy or results, you MUST call `search_filings` first and answer ONLY from \
the excerpts it returns. Never answer such questions from your own memory, even if \
you think you know the answer.
- If the excerpts do not contain the answer, say that the filings searched do not \
cover it. Do not fill the gap from memory.

Citations:
- Each excerpt starts with a header like [NVDA · Item 1A | FY ending 2026-01-25]. \
After every claim, cite the part before the "|", for example [NVDA · Item 1A].
- Companies' fiscal years end on different dates; mention the fiscal year when it \
matters.

Searching well:
- When comparing companies, search once per company using the `ticker` argument.
- For industry-wide questions, search without a ticker. You may rephrase and search \
again if results are thin, but make at most 4 searches per question in total.
- Before concluding that something is not in the filings, try at least one rephrased \
search (within the limit). If it is still not found, say so and stop; do not offer \
to search more.

Unclear questions:
- If a question refers to something that is not in the conversation (e.g. "them", \
"it", "that company" with nothing earlier to point to), ask the user what they mean \
instead of guessing and searching.

General questions:
- Questions that are not about specific companies (e.g. "what is a foundry?") may \
be answered from general knowledge without searching. Start such answers with \
"General knowledge (not from the filings):".

Be concise: a short direct answer first, then supporting details."""


# Hard limit enforced in code. The prompt asks for at most 4, but a prompt is only a
# request: one reply can ask for many parallel searches at once (we saw 46).
MAX_SEARCHES_PER_QUESTION = 6

BUDGET_NOTE = (
    "Search limit reached: this question would need more searches than the limit of "
    f"{MAX_SEARCHES_PER_QUESTION} per question. Do not search again. Answer from the "
    "excerpts already found, with citations. If they are not enough, explain that the "
    "question is too broad for one answer (not that you lack access to the filings) "
    "and suggest narrowing it, e.g. to up to 4 named companies."
)


def searches_for_current_question(messages: list[AnyMessage]) -> int:
    """Count tool results since the user's latest message."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            count += 1
    return count


class AgentState(TypedDict):
    """Everything the graph remembers while it runs.

    `add_messages` is a reducer: when a node returns {"messages": [new]}, the new
    messages are APPENDED to the list instead of replacing it, so the state holds
    the whole conversation: question, tool calls, tool results, answer.
    """

    messages: Annotated[list[AnyMessage], add_messages]


def build_agent(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Wire the agent graph around any chat model that supports tool calling.

    With a checkpointer, the state is saved after every step under the
    `thread_id` passed in the config, and the next invoke on the same thread
    continues from it, so follow-up questions see the earlier turns.
    """
    llm_with_tools = llm.bind_tools(TOOLS)
    # Same tools visible, but calling them is forbidden: used to force an answer.
    llm_must_answer = llm.bind_tools(TOOLS, tool_choice="none")

    def agent(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Node: ask the LLM what to do next, given the conversation so far."""
        # The system prompt is prepended on every call but never stored in the
        # state, so the state stays pure conversation (no duplicates per turn).
        prompt = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        used = searches_for_current_question(state["messages"])

        if used < MAX_SEARCHES_PER_QUESTION:
            reply = llm_with_tools.invoke(prompt)
            if used + len(reply.tool_calls) <= MAX_SEARCHES_PER_QUESTION:
                return {"messages": [reply]}  # appended by add_messages
        # Over budget (already, or this reply asked for too many searches at once):
        # discard any tool requests and make the model answer without tools.
        reply = llm_must_answer.invoke([*prompt, SystemMessage(BUDGET_NOTE)])
        return {"messages": [reply]}

    def route(state: AgentState) -> Literal["tools", "__end__"]:
        """Edge: if the LLM asked for tools, run them; otherwise we're done."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))  # runs each requested tool call
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")  # tool results go back to the LLM

    return graph.compile(checkpointer=checkpointer)


@cache
def get_agent(memory: bool = False) -> CompiledStateGraph:
    """The agent with the configured OpenAI model (built once per `memory` value).

    memory=False: one-shot questions (`ask`), every invoke starts fresh.
    memory=True: conversations kept in process memory; every invoke must pass
    config={"configurable": {"thread_id": ...}}.
    """
    settings = get_settings()
    llm = ChatOpenAI(model=settings.answer_model, api_key=settings.openai_api_key)
    return build_agent(llm, checkpointer=MemorySaver() if memory else None)
