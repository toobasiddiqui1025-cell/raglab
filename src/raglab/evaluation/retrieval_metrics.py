"""Retrieval evaluation metrics module.

Implements page-level retrieval metrics: Recall@k, Mean Reciprocal Rank (MRR),
and Normalized Discounted Cumulative Gain (nDCG@k). A retrieved chunk is deemed
a hit if its doc_id and page_number match a ground truth target.
"""

from typing import List, Set, Tuple


def recall_at_k(
    retrieved: List[Tuple[str, int]],
    gold_targets: Set[Tuple[str, int]],
    k: int = 5,
) -> float:
    """Compute Recall@k at the (doc_id, page_number) level.

    Args:
        retrieved: Ordered list of (doc_id, page_number) tuples retrieved.
        gold_targets: Set of ground-truth (doc_id, page_number) targets.
        k: Cutoff rank for evaluation.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """
    raise NotImplementedError


def mean_reciprocal_rank(
    retrieved: List[Tuple[str, int]],
    gold_targets: Set[Tuple[str, int]],
) -> float:
    """Compute Reciprocal Rank (RR) for the first matching hit.

    Args:
        retrieved: Ordered list of (doc_id, page_number) tuples retrieved.
        gold_targets: Set of ground-truth (doc_id, page_number) targets.

    Returns:
        float: Reciprocal of the 1-based rank of the first hit, or 0.0 if no hit.
    """
    raise NotImplementedError


def ndcg_at_k(
    retrieved: List[Tuple[str, int]],
    gold_targets: Set[Tuple[str, int]],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@k).

    Args:
        retrieved: Ordered list of (doc_id, page_number) tuples retrieved.
        gold_targets: Set of ground-truth (doc_id, page_number) targets.
        k: Cutoff rank for evaluation.

    Returns:
        float: nDCG@k score between 0.0 and 1.0.
    """
    raise NotImplementedError
