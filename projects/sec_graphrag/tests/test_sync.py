from sec_graphrag.evals.models import Category, EvalExample
from sec_graphrag.evals.sync import (
    CI_PER_CATEGORY,
    ci_questions,
    example_id,
    plan_sync,
    to_langsmith,
)


def _example(
    category: Category, number: int, answer: str = "An answer."
) -> EvalExample:
    return EvalExample(
        question=f"Question number {number} about {category}?",
        reference_answer=answer,
        category=category,
        source_chunks=["NVDA-1-0000"],
    )


def test_same_question_always_gets_the_same_id() -> None:
    a = _example(Category.lookup, 1)
    edited_answer = _example(Category.lookup, 1, answer="A better answer.")

    assert example_id(a) == example_id(edited_answer)
    assert example_id(a) != example_id(_example(Category.lookup, 2))


def test_ci_split_takes_the_same_number_from_every_category() -> None:
    examples = [_example(c, i) for c in Category for i in range(10)]

    ci = ci_questions(examples)

    assert len(ci) == CI_PER_CATEGORY * len(Category)
    assert ci == ci_questions(list(reversed(examples)))  # file order doesn't matter


def test_plan_creates_updates_deletes_and_skips_unchanged() -> None:
    kept, edited, new = (_example(Category.lookup, i) for i in range(3))
    gone = to_langsmith(_example(Category.lookup, 9), in_ci=False)
    remote = {
        example_id(kept): to_langsmith(kept, in_ci=False),
        example_id(edited): to_langsmith(edited, in_ci=False),
        gone["id"]: gone,
    }
    edited = edited.model_copy(update={"reference_answer": "Corrected."})
    local = [to_langsmith(e, in_ci=False) for e in (kept, edited, new)]

    plan = plan_sync(local, remote)

    assert [e["id"] for e in plan.create] == [example_id(new)]
    assert [e["id"] for e in plan.update] == [example_id(edited)]
    assert plan.delete == [gone["id"]]
    assert plan.unchanged == 1
