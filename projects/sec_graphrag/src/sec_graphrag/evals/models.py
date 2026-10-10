"""The eval dataset: what one test question looks like, and how to load the file.

`dataset.jsonl` (in this folder, tracked in git) is the source of truth: one JSON
object per line, one line per question. `eval sync` (step 2.4) uploads it to
LangSmith; we never edit the dataset there.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"


class Category(StrEnum):
    """The kind of reasoning a question needs. Scores are reported per category,
    because each retrieval method is expected to win in different ones."""

    # One fact from one filing: "Which foundry makes NVDA's GPUs?"
    lookup = "lookup"
    # Facts chained across filings: "Which of AMD's suppliers also supply NVDA?"
    multi_hop = "multi_hop"
    # Count or list over many filings: "How many companies depend on TSMC?"
    aggregation = "aggregation"
    # Summarise a theme: "What export-control risks do the companies share?"
    thematic = "thematic"


class EvalExample(BaseModel):
    """One eval question with everything needed to score an answer to it."""

    question: str = Field(min_length=10)
    # A short correct answer, written from the filing text (never from the graph,
    # so graph-based retrieval is not favoured). The LLM judge compares against it.
    reference_answer: str = Field(min_length=1)
    # Names that a good answer must mention, e.g. ["TSMC", "Samsung"]; scored by
    # the deterministic `entity_recall` evaluator.
    expected_entities: list[str] = Field(default_factory=list)
    category: Category
    # IDs of the chunks the reference answer was written from, e.g. "NVDA-1A-0007",
    # so every reference answer can be checked against the source text.
    source_chunks: list[str] = Field(min_length=1)


def load_dataset(path: Path = DATASET_PATH) -> list[EvalExample]:
    """Read and validate every line; fail with the line number of a bad one."""
    examples: list[EvalExample] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue  # allow blank lines
        try:
            examples.append(EvalExample.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"{path.name} line {number}: {error}") from error
    return examples
