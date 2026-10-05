"""`sec-graphrag doctor`: check that every external service is configured and reachable."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from sec_graphrag.config import get_settings
from sec_graphrag.db import get_driver


class Status(StrEnum):
    ok = "ok"
    warn = "warn"
    fail = "fail"


class CheckWarning(Exception):
    """Raised by a check for a problem that does not block the current milestone."""


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


def check_neo4j() -> str:
    driver = get_driver()
    driver.verify_connectivity()
    info = driver.get_server_info()
    return f"{info.agent} at {info.address}"


def check_openai_chat() -> str:
    settings = get_settings()
    llm = ChatOpenAI(model=settings.answer_model, api_key=settings.openai_api_key)
    # run_name and tags make this call easy to find in the LangSmith UI.
    reply = llm.invoke(
        "Reply with exactly: OK",
        config={"run_name": "doctor-chat-check", "tags": ["doctor"]},
    )
    return f"{settings.answer_model} replied {reply.text.strip()!r}"


def check_openai_embeddings() -> str:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    vector = embeddings.embed_query("ping")
    return f"{settings.embedding_model} returned {len(vector)} dimensions"


def check_langsmith() -> str:
    settings = get_settings()
    if settings.langsmith_api_key is None or not settings.langsmith_tracing:
        raise CheckWarning(
            "tracing is off (set LANGSMITH_API_KEY and LANGSMITH_TRACING)"
        )
    # Traces are sent in the background; make sure the chat check's trace has arrived.
    wait_for_all_tracers()
    project = Client().read_project(project_name=settings.langsmith_project)
    return f"project {project.name!r}: {project.url}"


def check_edgar_identity() -> str:
    identity = get_settings().edgar_identity
    if not identity:
        raise CheckWarning('EDGAR_IDENTITY not set (needed from M1: "Name email")')
    return identity


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("Neo4j", check_neo4j),
    ("OpenAI chat", check_openai_chat),
    ("OpenAI embeddings", check_openai_embeddings),
    ("LangSmith", check_langsmith),
    ("EDGAR identity", check_edgar_identity),
]


def run_checks() -> list[CheckResult]:
    """Run every check; one failing check never stops the others."""
    results = []
    for name, check in CHECKS:
        try:
            results.append(CheckResult(name, Status.ok, check()))
        except CheckWarning as w:
            results.append(CheckResult(name, Status.warn, str(w)))
        except Exception as e:  # noqa: BLE001 — a doctor must report any failure
            first_line = str(e).splitlines()[0] if str(e) else ""
            results.append(
                CheckResult(name, Status.fail, f"{type(e).__name__}: {first_line}")
            )
    return results
