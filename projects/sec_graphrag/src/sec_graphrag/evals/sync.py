"""Upload `dataset.jsonl` to the LangSmith dataset `sec-graphrag-qa` (step 2.4).

Git is the source of truth; LangSmith gets a copy that experiments run against.
`sync` makes LangSmith match the file: new questions are created, changed ones
updated, removed ones deleted, unchanged ones left alone. Every change makes a
new dataset *version* in LangSmith, so an old experiment still shows exactly
which questions it ran on.

Each LangSmith example has:
- inputs:   {"question"}                         -> what the agent receives
- outputs:  {"reference_answer", "expected_entities", "source_chunks"}
                                                 -> what evaluators compare with
- metadata: {"category"}                         -> scores per category
- split:    ["full"] or ["ci", "full"]           -> which subset it belongs to
"""

import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langsmith import Client

from sec_graphrag.evals.models import Category, EvalExample

DATASET_NAME = "sec-graphrag-qa"
DESCRIPTION = (
    "Questions over the 10-K filings of 22 semiconductor companies, with reference "
    "answers written from the filing text. Source of truth: "
    "projects/sec_graphrag/src/sec_graphrag/evals/dataset.jsonl"
)
CI_PER_CATEGORY = 3  # 4 categories x 3 = 12 questions run on every PR

# Fixed namespace for uuid5: the same question always gets the same example ID,
# so a re-sync finds and updates it instead of adding a duplicate.
_ID_NAMESPACE = uuid.UUID("5ec9a0c1-7b9e-4c35-9a2b-2d4d1f0e8a11")


def example_id(example: EvalExample) -> uuid.UUID:
    """Stable ID from the question text (editing the question makes a new one)."""
    return uuid.uuid5(_ID_NAMESPACE, example.question)


def ci_questions(examples: list[EvalExample], seed: int = 42) -> set[str]:
    """The small `ci` split: CI_PER_CATEGORY questions from every category, so
    a quick run still covers each kind of question."""
    by_category: dict[Category, list[str]] = defaultdict(list)
    for example in examples:
        by_category[example.category].append(example.question)
    rng = random.Random(seed)
    picked: set[str] = set()
    # Fixed category order and sorted questions: the sample depends only on the
    # seed, never on the order of lines in the file.
    for category in Category:
        questions = sorted(by_category[category])
        picked.update(rng.sample(questions, min(CI_PER_CATEGORY, len(questions))))
    return picked


def to_langsmith(example: EvalExample, in_ci: bool) -> dict[str, Any]:
    """One local example in the shape LangSmith stores."""
    return {
        "id": example_id(example),
        "inputs": {"question": example.question},
        "outputs": {
            "reference_answer": example.reference_answer,
            "expected_entities": example.expected_entities,
            "source_chunks": example.source_chunks,
        },
        "metadata": {"category": example.category.value},
        "split": ["ci", "full"] if in_ci else ["full"],  # sorted, for comparing
    }


@dataclass
class SyncPlan:
    create: list[dict[str, Any]] = field(default_factory=list)
    update: list[dict[str, Any]] = field(default_factory=list)
    delete: list[uuid.UUID] = field(default_factory=list)
    unchanged: int = 0


def plan_sync(
    local: list[dict[str, Any]], remote: dict[uuid.UUID, dict[str, Any]]
) -> SyncPlan:
    """Compare local examples with what LangSmith has (both in `to_langsmith`
    shape, keyed by ID) and decide what to send. No network calls: testable."""
    plan = SyncPlan()
    for example in local:
        current = remote.get(example["id"])
        if current is None:
            plan.create.append(example)
        elif current != example:
            plan.update.append(example)
        else:
            plan.unchanged += 1
    local_ids = {example["id"] for example in local}
    plan.delete = [id_ for id_ in remote if id_ not in local_ids]
    return plan


def _remote_shape(example: Any) -> dict[str, Any]:
    """A LangSmith example in `to_langsmith` shape, for comparison. LangSmith
    keeps the splits inside metadata as "dataset_split"."""
    metadata = dict(example.metadata or {})
    splits = metadata.pop("dataset_split", ["base"])
    return {
        "id": example.id,
        "inputs": example.inputs,
        "outputs": example.outputs,
        "metadata": {"category": metadata.get("category")},
        "split": sorted(splits),
    }


def sync(examples: list[EvalExample], client: Client | None = None) -> SyncPlan:
    """Make the LangSmith dataset match `examples`; returns what was done."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(DATASET_NAME, description=DESCRIPTION)

    ci = ci_questions(examples)
    local = [to_langsmith(e, e.question in ci) for e in examples]
    remote = {
        e.id: _remote_shape(e) for e in client.list_examples(dataset_id=dataset.id)
    }
    plan = plan_sync(local, remote)

    if plan.create:
        client.create_examples(dataset_id=dataset.id, examples=plan.create)
    if plan.update:
        client.update_examples(dataset_id=dataset.id, updates=plan.update)
    if plan.delete:
        client.delete_examples(plan.delete)
    return plan
