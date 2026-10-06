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
from langchain_core.messages import AIMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from sec_graphrag.agent.tools import TOOLS
from sec_graphrag.config import get_settings


class AgentState(TypedDict):
    """Everything the graph remembers while it runs.

    `add_messages` is a reducer: when a node returns {"messages": [new]}, the new
    messages are APPENDED to the list instead of replacing it, so the state holds
    the whole conversation: question, tool calls, tool results, answer.
    """

    messages: Annotated[list[AnyMessage], add_messages]


def build_agent(llm: BaseChatModel) -> CompiledStateGraph:
    """Wire the agent graph around any chat model that supports tool calling."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Node: ask the LLM what to do next, given the conversation so far."""
        reply = llm_with_tools.invoke(state["messages"])
        return {"messages": [reply]}  # appended by add_messages

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

    return graph.compile()


@cache
def get_agent() -> CompiledStateGraph:
    """The agent with the configured OpenAI model (built once)."""
    settings = get_settings()
    llm = ChatOpenAI(model=settings.answer_model, api_key=settings.openai_api_key)
    return build_agent(llm)
