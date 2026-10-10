from sec_graphrag.evals.models import Category, EvalExample
from sec_graphrag.evals.review import pick_balanced


def _example(category: Category, number: int) -> EvalExample:
    return EvalExample(
        question=f"Question number {number} about {category}?",
        reference_answer="An answer.",
        category=category,
        source_chunks=["NVDA-1-0000"],
    )


def test_fills_each_category_up_to_its_target() -> None:
    dataset = [_example(Category.lookup, 0)]  # 1 lookup already there
    candidates = [_example(Category.lookup, i) for i in range(1, 6)] + [
        _example(Category.thematic, i) for i in range(1, 3)
    ]
    targets = {Category.lookup: 3, Category.thematic: 5}

    picked = pick_balanced(dataset, candidates, targets)

    kinds = [p.category for p in picked]
    assert kinds.count(Category.lookup) == 2  # 3 wanted - 1 already there
    assert kinds.count(Category.thematic) == 2  # wanted 5, only 2 exist


def test_running_again_adds_nothing() -> None:
    candidates = [_example(Category.lookup, i) for i in range(5)]
    targets = {Category.lookup: 3}

    first = pick_balanced([], candidates, targets)

    assert pick_balanced(first, candidates, targets) == []
