from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .artifacts import build_artifacts, embed_text, expand_query, text_for_product, tokenize
from .loaders import load_products
from .models import Product, SearchResponse, SearchResult


class HybridDemoSearch:
    """Lightweight public demo hybrid search: BM25 + precomputed semantic vectors."""

    def __init__(
        self,
        products: List[Product],
        embeddings: np.ndarray,
        vocabulary: dict[str, int] | None = None,
    ):
        self.products = products
        self.embeddings = embeddings.astype(np.float32)
        self.vocabulary = vocabulary
        self.product_by_sku: Dict[int, Product] = {
            product.sku_id: product for product in products
        }
        self.sku_ids = [product.sku_id for product in products]
        self.tokenized_corpus = [tokenize(text_for_product(product)) for product in products]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_paths(cls, catalog_path: Path, artifact_dir: Path) -> "HybridDemoSearch":
        products = load_products(catalog_path)
        embeddings_path = artifact_dir / "product_embeddings.npy"
        if not embeddings_path.exists():
            build_artifacts(products, artifact_dir)
        embeddings = np.load(embeddings_path)
        metadata_path = artifact_dir / "embedding_metadata.json"
        vocabulary = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            vocabulary = metadata.get("vocabulary")
        return cls(products=products, embeddings=embeddings, vocabulary=vocabulary)

    def _bm25_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        tokens = tokenize(expand_query(query))
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            (self.sku_ids[index], float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]

    def _semantic_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        query_embedding = embed_text(query, vocabulary=self.vocabulary)
        if np.linalg.norm(query_embedding) == 0:
            return []
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            (self.sku_ids[index], float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]

    @staticmethod
    def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            return {sku_id: 1.0 for sku_id in scores}
        return {
            sku_id: (score - min_score) / (max_score - min_score)
            for sku_id, score in scores.items()
        }

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        if len(query.strip()) < 2:
            raise ValueError("Query must contain at least 2 characters")

        top_k = max(1, min(top_k, 25))
        start = time.perf_counter()

        bm25_results = self._bm25_search(query, limit=top_k * 4)
        semantic_results = self._semantic_search(query, limit=top_k * 4)
        bm25_map = dict(bm25_results)
        semantic_map = dict(semantic_results)
        bm25_norm = self._normalize(bm25_map)
        semantic_norm = self._normalize(semantic_map)

        all_skus = set(bm25_map) | set(semantic_map)
        ranked = []
        for sku_id in all_skus:
            bm25_score = bm25_norm.get(sku_id, 0.0)
            semantic_score = semantic_norm.get(sku_id, 0.0)
            final_score = 0.45 * bm25_score + 0.55 * semantic_score
            ranked.append((sku_id, final_score, bm25_score, semantic_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        results = []
        for sku_id, final_score, bm25_score, semantic_score in ranked[:top_k]:
            product = self.product_by_sku[sku_id]
            if bm25_score > 0 and semantic_score > 0:
                source = "hybrid"
                explanation = "Matched by product terms and semantic similarity."
            elif semantic_score > 0:
                source = "semantic"
                explanation = "Matched by semantic similarity in the public demo catalog."
            else:
                source = "bm25"
                explanation = "Matched by exact or near-exact product terms."

            results.append(
                SearchResult(
                    sku_id=sku_id,
                    sku_name=product.sku_name,
                    final_score=round(final_score, 4),
                    bm25_score=round(bm25_score, 4),
                    semantic_score=round(semantic_score, 4),
                    source=source,
                    explanation=explanation,
                )
            )

        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(
            query=query,
            demo_mode=True,
            retrieval_mode="bm25+semantic",
            results=results,
            total_found=len(ranked),
            latency_ms=round(latency_ms, 2),
            notes=(
                "Public portfolio demo using a sanitized sample catalog. "
                "Product embeddings are precomputed; retrieval and ranking run live per request."
            ),
        )
