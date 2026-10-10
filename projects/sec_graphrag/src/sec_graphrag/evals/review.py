"""Move drafted candidates into the dataset (step 2.3).

The spec's step is a human review: accept, edit or reject each candidate after
checking it against its source chunks. For this learning project we skip that
and treat every candidate as correct (decided 2026-10-10), so "review" here only
picks a balanced sample: the dataset should have about 60 questions in fixed
proportions per category, not all 115 candidates.
"""

import random
from collections import Counter

from sec_graphrag.evals.models import Category, EvalExample

# How many questions of each category the dataset should end up with (spec §6).
TARGETS = {
    Category.lookup: 15,
    Category.multi_hop: 25,
    Category.aggregation: 10,
    Category.thematic: 10,
}


def pick_balanced(
    dataset: list[EvalExample],
    candidates: list[EvalExample],
    targets: dict[Category, int] = TARGETS,
    seed: int = 42,
) -> list[EvalExample]:
    """Candidates to add so each category reaches its target.

    Questions already in the dataset count toward the target and are never
    added twice, so running this again adds nothing once the targets are met.
    """
    have = Counter(example.category for example in dataset)
    known = {example.question for example in dataset}
    rng = random.Random(seed)
    picked: list[EvalExample] = []
    for category, target in targets.items():
        pool = [
            c for c in candidates if c.category is category and c.question not in known
        ]
        # Sort first so the sample depends only on the seed, not on file order.
        pool.sort(key=lambda c: c.question)
        missing = max(0, target - have[category])
        picked += rng.sample(pool, min(missing, len(pool)))
    return picked
