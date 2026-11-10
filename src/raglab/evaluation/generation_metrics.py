"""LLM-as-a-judge generation evaluation metrics module.

Evaluates generated answers on two key dimensions using strict rubric prompts:
1. Correctness: Evaluates whether the answer matches the reference answer.
2. Faithfulness: Evaluates whether every claim in the answer is grounded in retrieved context.
Caches judgments on disk keyed by SHA-256 hashes of inputs to conserve API quota.
"""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
from raglab.generation.llm import LLMClient


class JudgeResult(BaseModel):
    """Structured verdict from the LLM judge.

    Attributes:
        score: Numerical score (e.g., 0.0 to 1.0).
        reason: Justification explaining the score according to the rubric.
    """
    score: float
    reason: str


class GenerationEvaluator:
    """Evaluates generation quality using LLM-as-a-judge with local response caching."""

    def __init__(self, llm_client: LLMClient, cache_dir: Optional[Path] = None) -> None:
        """Initialize evaluator with LLM client and optional cache directory.

        Args:
            llm_client: Client used to execute rubric evaluation prompts.
            cache_dir: Optional directory to cache verdicts on disk.
        """
        pass

    def evaluate_correctness(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
    ) -> JudgeResult:
        """Score answer correctness against reference answer using strict rubric.

        Args:
            question: Original question.
            reference_answer: Ground truth reference answer.
            generated_answer: Model generated answer to grade.

        Returns:
            JudgeResult: Graded score and rationale.
        """
        raise NotImplementedError

    def evaluate_faithfulness(
        self,
        generated_answer: str,
        context_chunks: List[str],
    ) -> JudgeResult:
        """Score answer faithfulness against the provided context chunks.

        Args:
            generated_answer: Model generated answer.
            context_chunks: Chunks provided to the model during generation.

        Returns:
            JudgeResult: Faithfulness score (0.0 to 1.0) and rationale.
        """
        raise NotImplementedError
