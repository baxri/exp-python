"""A multi-turn chat session with the agent, streamed as simple events.

The session owns one `thread_id`: every turn uses it, so the agent's checkpointer
gives it the earlier turns (memory). `new_conversation()` starts a fresh thread.

`send()` yields events instead of printing, so the CLI decides how to show them
(and a web UI could reuse this unchanged):
    Search(tool, args)   the agent decided to run a tool
    Token(text)          a piece of the answer as the model generates it
    TurnDone(...)        the turn finished: sources and timing
"""

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from sec_graphrag.agent.graph import get_agent
from sec_graphrag.retrieval.base import Source


@dataclass
class Token:
    text: str


@dataclass
class Search:
    tool: str
    args: dict[str, object]


@dataclass
class TurnDone:
    seconds: float
    sources: list[Source] = field(default_factory=list)


Event = Token | Search | TurnDone


class ChatSession:
    def __init__(self) -> None:
        self.agent = get_agent(memory=True)
        self.thread_id = ""
        self.new_conversation()

    def new_conversation(self) -> None:
        """Forget the conversation: a new thread_id means an empty history."""
        self.thread_id = str(uuid.uuid4())

    def warm_up(self) -> None:
        """Connect to Neo4j now (~2-3s), so the first question doesn't pay for it."""
        from sec_graphrag.retrieval.vector import _store

        _store()

    def send(self, text: str) -> Iterator[Event]:
        """Send one user message and stream the agent's reaction as events.

        This is a generator (it uses `yield`): the caller gets each event the
        moment it happens, e.g. `for event in session.send("..."): print(event)`.
        """
        # Remember when the turn started, to report how long it took.
        start = time.perf_counter()

        # Settings for this one run of the graph.
        config = {
            # thread_id tells the checkpointer WHICH conversation to load and save.
            # Same id as earlier turns -> the agent sees the earlier messages.
            "configurable": {"thread_id": self.thread_id},
            # The same id as metadata: LangSmith groups all turns with the same
            # thread_id into one "thread" in its UI.
            "metadata": {"thread_id": self.thread_id},
            # Name of this turn's root run in LangSmith.
            "run_name": "chat_turn",
            # Label for filtering chat runs in LangSmith.
            "tags": ["chat"],
        }

        # Run the graph and receive its output piece by piece while it runs.
        # Two stream modes at once, each item says which mode it came from:
        #   "messages": every token the LLM generates, as soon as it exists
        #   "updates":  each node's complete output once that node finishes
        for mode, payload in self.agent.stream(
            # Input: just the new user message. The checkpointer adds the
            # earlier turns of this thread before the agent runs.
            {"messages": [("user", text)]},
            # The settings built above (thread_id, metadata, name, tags).
            config=config,
            # Ask for both kinds of output.
            stream_mode=["messages", "updates"],
        ):
            # Case 1: a token (small piece of text) from an LLM call.
            if mode == "messages":
                # payload = (the token chunk, info about where it came from).
                chunk, meta = payload
                # Keep only answer text written by the `agent` node:
                if (
                    # from the agent node (not the tool results of `tools`),
                    meta.get("langgraph_node") == "agent"
                    # a piece of an AI message being generated,
                    and isinstance(chunk, AIMessageChunk)
                    # with actual text (tool-request chunks have empty text).
                    and chunk.text
                ):
                    # Hand the token to the caller right away -> live typing.
                    yield Token(chunk.text)

            # Case 2: a node finished; payload = {node name: what it returned}.
            elif mode == "updates":
                # Usually one node per update, but it's a dict, so loop.
                for node, update in payload.items():
                    # Only the agent node can request searches; skip the rest
                    # (and empty updates).
                    if node != "agent" or not update:
                        continue
                    # The agent node returns {"messages": [its reply]}.
                    for message in update["messages"]:
                        # Its reply is an AIMessage, possibly with tool calls.
                        if isinstance(message, AIMessage):
                            # One event per requested tool call ->
                            # the CLI shows "🔎 search_filings(...)".
                            for call in message.tool_calls:
                                yield Search(call["name"], call["args"])

        # The stream ended: the agent answered without asking for more tools.
        # Report the duration and the sources this turn's searches found.
        yield TurnDone(round(time.perf_counter() - start, 1), self._turn_sources())

    def _turn_sources(self) -> list[Source]:
        """Sources found by the searches of the latest turn (each chunk once)."""
        messages = self.agent.get_state(
            {"configurable": {"thread_id": self.thread_id}}
        ).values["messages"]
        sources: dict[str, Source] = {}
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break  # reached the start of this turn
            if isinstance(message, ToolMessage) and message.artifact:
                for item in message.artifact:
                    # State read back from the checkpointer has been serialized:
                    # artifacts may come back as plain dicts, so rebuild Sources.
                    source = Source.model_validate(item)
                    sources.setdefault(source.chunk_id, source)
        return list(sources.values())
