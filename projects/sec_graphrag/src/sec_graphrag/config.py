"""Typed application settings, loaded once from environment variables and `.env` files."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> sec_graphrag/ -> src/ -> projects/sec_graphrag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Later files override earlier ones; real environment variables override both.
        env_file=(REPO_ROOT / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # The shared .env holds variables for other scripts too; don't fail on them.
        extra="ignore",
    )

    # Neo4j (required: no default, so a missing value fails at startup)
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: SecretStr
    neo4j_database: str = "neo4j"

    # OpenAI
    openai_api_key: SecretStr

    # LangSmith (optional until step 0.5)
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "sec-graphrag"

    # SEC EDGAR requires a "Name email" identity on every request (needed from M1)
    edgar_identity: str | None = None

    # Models
    extract_model: str = "gpt-5-mini"
    answer_model: str = "gpt-5-mini"
    judge_model: str = "gpt-5.2"
    embedding_model: str = "text-embedding-3-small"

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment only on the first call."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
