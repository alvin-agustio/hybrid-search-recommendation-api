"""Hybrid search pipeline combining BM25 and semantic search."""

import time
from dataclasses import dataclass
from typing import List, Tuple

from inference.bm25 import BM25Baseline
from inference.post_processing import FusedResult, ScoreFusion


@dataclass
class HybridSearchResult:
    """Result from hybrid search pipeline."""

    sku_id: int
    score: float
    name: str = ""
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    reranked: bool = False


@dataclass
class SearchResult:
    """Search result with products and latency metadata."""

    products: List[dict]
    latency_ms: float


class HybridSearchPipeline:
    """Hybrid search combining BM25 keyword matching and semantic similarity."""

    def __init__(
        self,
        bm25: BM25Baseline,
        semantic_searcher=None,
        confidence_threshold: float = 10.0,
        bm25_confidence_threshold: float = 15.0,  # Tuned for Word-Only BM25 (was 65.0 for n-grams)
        bm25_score_floor: float = 0.0,  # Keep disabled so short exact queries are not discarded prematurely.
        bm25_weight: float = 0.4,  # BM25 weight in score fusion
        semantic_weight: float = 0.6,  # Semantic weight in score fusion
        rerank_mode: str = "score_fusion",  # "score_fusion" | "semantic_rerank"
        # Backward compat: accept but ignore old params
        fusion_k: int = 60,  # deprecated, kept for API compat
    ):
        """Initialize hybrid search pipeline.

        Args:
            bm25_score_floor: Minimum BM25 score for a result to be included in fusion.
                Results below this are discarded before fusion. Default=25.0.
                Set to 0.0 to disable (original behaviour).
            bm25_weight: Weight for normalized BM25 scores in fusion. Default=0.4.
            semantic_weight: Weight for semantic (cosine) scores in fusion. Default=0.6.
            rerank_mode: Fusion strategy.
                'score_fusion' (default): Weighted normalized score fusion.
                'semantic_rerank': BM25 expands candidates, semantic scores determine final rank.
        """
        # Store search components
        self.bm25 = bm25
        self.semantic_searcher = semantic_searcher
        self.fusion = ScoreFusion(
            bm25_weight=bm25_weight, semantic_weight=semantic_weight
        )
        self.confidence_threshold = confidence_threshold
        self.bm25_confidence_threshold = bm25_confidence_threshold
        self.bm25_score_floor = bm25_score_floor
        self.rerank_mode = rerank_mode
        self.fusion_k = fusion_k

        # Build product name lookup from BM25
        self.product_names = {}
        if hasattr(bm25, "product_names") and hasattr(bm25, "sku_ids"):
            self.product_names = dict(zip(bm25.sku_ids, bm25.product_names))

    def _detect_intent_weights(self, query: str) -> Tuple[float, float]:
        """
        Dynamically determine fusion weights based on lexical frequency intent.

        Uses Document Frequency (DF) of query words in the BM25 index:
        - Conceptual Intent: Words rarely/never appear (e.g., "sarapan" DF=0)
          -> Boost semantic weight to 0.9
        - Exact/Brand Intent: All words are common (e.g., "susu ultra" min DF=133)
          -> Boost BM25 weight to 0.7
        - Mixed/Default: Keep configured weights.
        """
        if not hasattr(self.bm25, "get_word_frequencies"):
            return self.fusion.bm25_weight, self.fusion.semantic_weight

        freqs = self.bm25.get_word_frequencies(query)
        if not freqs:
            return self.fusion.bm25_weight, self.fusion.semantic_weight

        max_freq = max(freqs)
        min_freq = min(freqs)

        # 1. Conceptual Intent (Words don't exist in catalog, or very rare)
        # e.g., "sarapan" (min=0, max=0), "cemilan malam" (min=5, max=7)
        if min_freq == 0 and max_freq < 50:
            return 0.1, 0.9  # 90% Semantic

        # 2. Exact/Brand Intent (All words are well-known in catalog)
        # e.g., "susu ultra" (min=133), "teh botol" (min=204)
        if min_freq > 10:
            return 0.7, 0.3  # 70% BM25

        # 3. Default (Mixed or unclear)
        return self.fusion.bm25_weight, self.fusion.semantic_weight

    def search(
        self,
        query: str,
        top_k: int = 20,
        use_semantic: bool = True,
        debug_timing: bool = False,
    ) -> SearchResult:
        """Execute hybrid search with conditional semantic search."""
        # Validate query length
        if len(query.strip()) < 2:
            raise ValueError("Query too short, minimum 2 characters required")

        # Validate query pattern - detect repeated characters
        import re

        cleaned_query = query.strip().lower()
        # Check for same character repeated 4+ times consecutively (stricter pattern)
        if re.search(r"(.)\1{3,}", cleaned_query):
            raise ValueError("Invalid query pattern: repeated characters detected")

        # Start timing
        start_time = time.time()
        timings = {}

        # Run BM25 search (2x top_k for better fusion)
        # We explicitly disable SymSpell here since Semantic layer handles typo tolerance perfectly
        t0 = time.time()

        bm25_results = self.bm25.search(query, top_k=top_k * 2, use_symspell=False)
        timings["bm25"] = (time.time() - t0) * 1000

        # Filter BM25 results below score floor
        if self.bm25_score_floor > 0:
            bm25_results = [
                (sku_id, score)
                for sku_id, score in bm25_results
                if score >= self.bm25_score_floor
            ]

        # Dynamic Intent Detection for fusion weights
        w_bm25, w_sem = self._detect_intent_weights(query)
        dynamic_fusion = ScoreFusion(bm25_weight=w_bm25, semantic_weight=w_sem)

        # Conditional semantic search based on BM25 confidence
        semantic_results = []
        semantic_skipped = False
        routing_decision = "hybrid"  # default

        # SCORE-BASED ROUTING
        if use_semantic and self.semantic_searcher is not None:
            if bm25_results:
                top_score = bm25_results[0][1]

                if top_score >= self.bm25_confidence_threshold and w_bm25 >= w_sem:
                    # BM25 CONFIDENT and intent is Exact/Hybrid: High score
                    # Safe to skip semantic search to save heavy compute latency
                    timings["semantic"] = 0
                    semantic_skipped = True
                    routing_decision = "bm25_only"
                else:
                    # BM25 WEAK or intent is Conceptual (w_sem > w_bm25): Get semantic help
                    t1 = time.time()
                    semantic_results = self.semantic_searcher.search(
                        query, top_k=top_k * 2
                    )
                    timings["semantic"] = (time.time() - t1) * 1000
                    routing_decision = "hybrid"
            else:
                # No BM25 results
                t1 = time.time()
                semantic_results = self.semantic_searcher.search(query, top_k=top_k * 2)
                timings["semantic"] = (time.time() - t1) * 1000
                routing_decision = "semantic_only"
        else:
            timings["semantic"] = 0
            semantic_skipped = True
            routing_decision = "bm25_only"

        # Fuse results based on routing decision
        t2 = time.time()
        if routing_decision == "hybrid" and semantic_results:
            if self.rerank_mode == "semantic_rerank":
                # Semantic Rerank mode
                bm25_map = {sku_id: score for sku_id, score in bm25_results}
                sem_map = {sku_id: score for sku_id, score in semantic_results}
                all_skus = set(bm25_map.keys()) | set(sem_map.keys())
                ranked = sorted(
                    all_skus, key=lambda s: sem_map.get(s, 0.0), reverse=True
                )
                fused = [
                    FusedResult(
                        sku_id=s,
                        score=sem_map.get(s, 0.0),
                        bm25_score=bm25_map.get(s, 0.0),
                        semantic_score=sem_map.get(s, 0.0),
                    )
                    for s in ranked
                ]
            else:
                # Weighted Score Fusion using Dynamic Weights
                fused = dynamic_fusion.fuse(bm25_results, semantic_results)
        elif routing_decision == "semantic_only" and semantic_results:
            # SEMANTIC ONLY
            fused = [
                FusedResult(
                    sku_id=sku_id,
                    score=w_sem * score,
                    bm25_score=0.0,
                    semantic_score=score,
                )
                for sku_id, score in semantic_results
            ]
        else:
            # BM25 ONLY
            fused = [
                FusedResult(
                    sku_id=sku_id,
                    score=w_bm25
                    * score,  # Although relative ranking holds, maintain proper score scale
                    bm25_score=score,
                    semantic_score=0.0,
                )
                for sku_id, score in bm25_results
            ]
        timings["fusion"] = (time.time() - t2) * 1000

        # Limit to top_k results
        fused = fused[:top_k]

        # Convert to HybridSearchResult with product names
        t3 = time.time()
        results = [
            HybridSearchResult(
                sku_id=f.sku_id,
                score=f.score,
                name=self.product_names.get(f.sku_id, ""),
                bm25_score=f.bm25_score,
                semantic_score=f.semantic_score,
                reranked=False,
            )
            for f in fused
        ]
        timings["result_conversion"] = (time.time() - t3) * 1000

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Convert to API response format
        t4 = time.time()
        products = [
            {
                "sku_id": r.sku_id,
                "sku_name": r.name,
                "fused_score": r.score,
                "bm25_score": r.bm25_score,
                "semantic_score": r.semantic_score,
                "reranked": r.reranked,
            }
            for r in results
        ]
        timings["api_format"] = (time.time() - t4) * 1000

        # Debug timing output
        if debug_timing or semantic_skipped:
            import logging

            logger = logging.getLogger(__name__)
            path_type = (
                "FAST PATH (semantic skipped)"
                if semantic_skipped
                else "SLOW PATH (semantic used)"
            )
            logger.info(
                "%s | Query: '%s' | BM25=%.1fms, Semantic=%.1fms, Fusion=%.1fms, Convert=%.1fms, Format=%.1fms, Total=%.1fms",
                path_type,
                query,
                timings["bm25"],
                timings["semantic"],
                timings["fusion"],
                timings["result_conversion"],
                timings["api_format"],
                total_latency_ms,
            )

        # Return results with latency
        return SearchResult(
            products=products,
            latency_ms=total_latency_ms,
        )
