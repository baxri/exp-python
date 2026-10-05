"""LangSmith tracing setup."""

import os

from sec_graphrag.config import Settings


def configure_langsmith(settings: Settings) -> bool:
    """Export LangSmith settings to `os.environ`, where LangChain looks for them.

    pydantic-settings reads `.env` but never writes to the process environment, so
    without this LangChain would not see the values. Returns whether tracing is on.

    Call this at program start, before any LangChain/LangSmith code runs: LangSmith
    caches environment lookups, so values set after its first lookup are ignored.
    """
    enabled = settings.langsmith_tracing and settings.langsmith_api_key is not None
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if settings.langsmith_api_key is not None:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return enabled
