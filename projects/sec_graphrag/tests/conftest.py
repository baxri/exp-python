"""Shared pytest fixtures. pytest loads this file automatically."""

from collections.abc import Callable

import pytest

from sec_graphrag.config import Settings

REQUIRED_ENV = {
    "NEO4J_URI": "neo4j+s://test.example:7687",
    "NEO4J_USERNAME": "test-user",
    "NEO4J_PASSWORD": "test-password",
    "OPENAI_API_KEY": "sk-test",
}

# Variables a developer may have exported in their shell; tests must not see them.
OPTIONAL_ENV = [
    "NEO4J_DATABASE",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "EDGAR_IDENTITY",
    "EXTRACT_MODEL",
    "ANSWER_MODEL",
    "JUDGE_MODEL",
    "EMBEDDING_MODEL",
]


@pytest.fixture
def make_settings(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Return a factory that builds Settings from a clean, fake environment.

    `.env` files are ignored, so tests never depend on (or leak) real credentials.
    Keyword arguments become extra environment variables, e.g. ANSWER_MODEL="x".
    """
    for name in OPTIONAL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)

    def factory(**env: str) -> Settings:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return Settings(_env_file=None)  # type: ignore[call-arg]

    return factory
