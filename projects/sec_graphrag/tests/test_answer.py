from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sec_graphrag.agent.answer import ANSWER_PROMPT, build_answer_chain


def test_prompt_puts_rules_in_system_and_inputs_in_human_message() -> None:
    messages = ANSWER_PROMPT.format_messages(
        question="Who makes NVIDIA's chips?",
        context="[NVDA · Item 1 | FY ending 2026-01-25]\nWe use TSMC.",
    )

    assert [m.type for m in messages] == ["system", "human"]
    assert "ONLY the excerpts" in messages[0].text
    assert "Who makes NVIDIA's chips?" in messages[1].text
    assert "We use TSMC." in messages[1].text


def test_chain_returns_the_model_text() -> None:
    # A fake model replies with canned text: no network, no cost.
    fake_llm = FakeListChatModel(responses=["TSMC makes them [NVDA · Item 1]."])
    chain = build_answer_chain(fake_llm)

    answer = chain.invoke({"question": "Who makes the chips?", "context": "..."})

    assert answer == "TSMC makes them [NVDA · Item 1]."
