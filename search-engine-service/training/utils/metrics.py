"""Evaluation metrics for search/recommendation systems."""

from typing import List, Set

import numpy as np


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K."""

    def dcg(scores: List[float]) -> float:
        """Compute DCG from relevance scores."""
        return sum(
            score / np.log2(i + 2)  # log2(i+2) because i is 0-indexed
            for i, score in enumerate(scores)
        )

    # Relevance scores (binary: 1 if relevant, 0 otherwise)
    relevance_scores = [1.0 if item in relevant else 0.0 for item in retrieved[:k]]

    # Ideal DCG (all relevant items at top)
    ideal_scores = sorted(relevance_scores, reverse=True)

    # Compute DCG and IDCG
    dcg_score = dcg(relevance_scores)
    idcg_score = dcg(ideal_scores)

    # Avoid division by zero
    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score
