"""Turn retrieved context into a cited answer.

    question + RetrievalResult.context  ──prompt──▶  LLM  ──▶  answer text

The chain is built with LCEL: `prompt | llm | parser`. Each step's output is the
next step's input, and LangSmith traces every step as a nested span.
"""

from functools import cache

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from sec_graphrag.config import get_settings

# The fixed rules. Kept in the system message so a question can't easily
# override them, and so OpenAI can cache this identical prefix across calls.
SYSTEM_PROMPT = """\
You answer questions about semiconductor companies using excerpts from their \
latest annual reports (SEC Form 10-K).

Rules:
- Use ONLY the excerpts in the context. Do not use outside knowledge.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Each excerpt starts with a header like [NVDA · Item 1A | FY ending 2026-01-25].
  After every claim, cite its source using the part before the "|", for
  example [NVDA · Item 1A]. Cite several sources if several support a claim.
- Companies' fiscal years end on different dates; when it matters, say which
  fiscal year a fact refers to.
- Be concise: a short direct answer first, then supporting details."""

# The parts that change every call: the question and the retrieved context.
HUMAN_PROMPT = """\
Question: {question}

Context:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)


def build_answer_chain(llm: BaseChatModel) -> Runnable[dict[str, str], str]:
    """prompt -> llm -> plain text. Takes {"question": ..., "context": ...}."""
    return ANSWER_PROMPT | llm | StrOutputParser()


@cache
def get_answer_chain() -> Runnable[dict[str, str], str]:
    """The answer chain with the configured OpenAI model (built once)."""
    settings = get_settings()
    llm = ChatOpenAI(model=settings.answer_model, api_key=settings.openai_api_key)
    return build_answer_chain(llm)
