"""Post-processing utilities for hybrid search result fusion."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FusedResult:
    """Fused search result with combined scores."""

    sku_id: int
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0


class ScoreFusion:
    """Weighted Score Fusion — combines BM25 and semantic using normalized scores.

    Unlike RRF (rank-based), this uses the actual score values:
    - BM25 scores are min-max normalized per-query to [0, 1]
    - Semantic scores are already cosine similarity in [0, 1]
    - Final: bm25_weight * norm_bm25 + semantic_weight * semantic_score

    This is self-adaptive:
    - Strong BM25 matches (high spread) → BM25 contribution rises naturally
    - Noisy BM25 matches (low/tight scores) → normalization compresses → semantic wins
    - No hardcoded brand lists, extra thresholds, or magic numbers
    """

    def __init__(self, bm25_weight: float = 0.4, semantic_weight: float = 0.6):
        """Initialize score fusion.

        Args:
            bm25_weight: Weight for normalized BM25 scores. Default=0.4.
            semantic_weight: Weight for semantic (cosine) scores. Default=0.6.
            Weights don't need to sum to 1 — relative magnitude matters.
        """
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def _normalize_bm25(
        self, results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Min-max normalize BM25 scores to [0, 1] within this result batch.

        If all scores are identical (max == min), all normalize to 0.5.
        """
        if not results:
            return []

        scores = [s for _, s in results]
        max_score = max(scores)
        min_score = min(scores)
        spread = max_score - min_score

        if spread < 1e-9:
            # All scores identical → uniform 0.5
            return [(sku_id, 0.5) for sku_id, _ in results]

        return [(sku_id, (score - min_score) / spread) for sku_id, score in results]

    def fuse(
        self,
        bm25_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
    ) -> List[FusedResult]:
        """Fuse BM25 and semantic results using weighted normalized scores.

        Products appearing in both lists get boosted (both contributions add up).
        Products appearing in only one list get that source's weighted score.
        """
        scores: Dict[int, FusedResult] = {}

        # Normalize BM25 scores to [0, 1]
        bm25_normalized = self._normalize_bm25(bm25_results)
        bm25_raw_map = {sku_id: raw for sku_id, raw in bm25_results}

        # Process normalized BM25 results
        for sku_id, norm_score in bm25_normalized:
            scores[sku_id] = FusedResult(
                sku_id=sku_id,
                score=self.bm25_weight * norm_score,
                bm25_score=bm25_raw_map.get(sku_id, 0.0),
                semantic_score=0.0,
            )

        # Process semantic results (already 0-1 cosine similarity)
        for sku_id, sem_score in semantic_results:
            if sku_id in scores:
                # Product in BOTH → add semantic contribution (mutual boost)
                scores[sku_id].score += self.semantic_weight * sem_score
                scores[sku_id].semantic_score = sem_score
            else:
                # Product only in semantic
                scores[sku_id] = FusedResult(
                    sku_id=sku_id,
                    score=self.semantic_weight * sem_score,
                    bm25_score=0.0,
                    semantic_score=sem_score,
                )

        # Sort by fused score (highest first)
        return sorted(scores.values(), key=lambda x: x.score, reverse=True)


# Backward-compatible alias
RRFusion = ScoreFusion
