"""Gold evaluation dataset loader and schema definition.

Manages loading, validation, and serialization of ground truth question-answer
benchmarks used to evaluate retrieval recall and answer faithfulness.
"""

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel


class GoldQuestion(BaseModel):
    """Represents a benchmark question with verified ground-truth sources.

    Attributes:
        id: Unique identifier (e.g. 'q001').
        type: Category of question ('factoid', 'multi_hop', 'comparative', 'unanswerable').
        question: Query text.
        reference_answer: Ground truth reference answer.
        gold_doc_id: Expected source document identifier.
        gold_pages: List of 1-indexed pages containing the evidence.
    """
    id: str
    type: Literal["factoid", "multi_hop", "comparative", "unanswerable"]
    question: str
    reference_answer: str
    gold_doc_id: str
    gold_pages: List[int]


def load_gold_dataset(dataset_path: Path) -> List[GoldQuestion]:
    """Load and validate benchmark questions from a JSON file.

    Args:
        dataset_path: Path to gold questions JSON file.

    Returns:
        List[GoldQuestion]: Validated list of benchmark questions.
    """
    raise NotImplementedError
