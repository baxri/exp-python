import os
from collections.abc import Callable

from sec_graphrag.config import Settings
from sec_graphrag.tracing import configure_langsmith


def test_tracing_off_without_api_key(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(LANGSMITH_TRACING="true")

    assert configure_langsmith(settings) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_exports_settings_to_environment(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        LANGSMITH_TRACING="true",
        LANGSMITH_API_KEY="lsv2-test",
        LANGSMITH_PROJECT="my-project",
    )

    assert configure_langsmith(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2-test"
    assert os.environ["LANGSMITH_PROJECT"] == "my-project"
