from collections.abc import Callable

import pytest
from pydantic import ValidationError

from sec_graphrag.config import Settings


def test_reads_required_values_from_environment(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings()

    assert settings.neo4j_uri == "neo4j+s://test.example:7687"
    assert settings.neo4j_password.get_secret_value() == "test-password"


def test_secrets_are_masked_when_printed(
    make_settings: Callable[..., Settings],
) -> None:
    text = repr(make_settings())

    assert "test-password" not in text
    assert "sk-test" not in text


def test_defaults_apply_when_not_set(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()

    assert settings.neo4j_database == "neo4j"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.langsmith_api_key is None


def test_environment_overrides_defaults(make_settings: Callable[..., Settings]) -> None:
    assert make_settings(ANSWER_MODEL="gpt-5.2").answer_model == "gpt-5.2"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_boolean_strings_are_parsed(
    make_settings: Callable[..., Settings], raw: str, expected: bool
) -> None:
    assert make_settings(LANGSMITH_TRACING=raw).langsmith_tracing is expected


def test_missing_required_values_fail_with_all_names(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings()  # sets up the clean environment
    monkeypatch.delenv("NEO4J_PASSWORD")
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]

    missing = {error["loc"][0] for error in exc_info.value.errors()}
    assert missing == {"neo4j_password", "openai_api_key"}
