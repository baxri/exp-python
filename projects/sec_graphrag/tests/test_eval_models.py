import json
from pathlib import Path

import pytest

from sec_graphrag.evals.models import Category, load_dataset

GOOD = {
    "question": "Which foundry manufactures NVIDIA's GPUs?",
    "reference_answer": "TSMC (and Samsung) manufacture them.",
    "expected_entities": ["TSMC"],
    "category": "lookup",
    "source_chunks": ["NVDA-1A-0012"],
}


def _write(path: Path, *rows: dict) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")
    return path


def test_loads_valid_lines_and_skips_blank_ones(tmp_path: Path) -> None:
    examples = load_dataset(_write(tmp_path / "d.jsonl", GOOD, GOOD))

    assert len(examples) == 2
    assert examples[0].category is Category.lookup


def test_bad_line_reports_its_number(tmp_path: Path) -> None:
    bad = {**GOOD, "category": "trivia"}  # not one of the four categories

    with pytest.raises(ValueError, match="line 2"):
        load_dataset(_write(tmp_path / "d.jsonl", GOOD, bad))


def test_reference_must_point_to_source_chunks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_chunks"):
        load_dataset(_write(tmp_path / "d.jsonl", {**GOOD, "source_chunks": []}))
