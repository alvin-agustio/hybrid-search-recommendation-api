"""FAISS index wrapper for fast product similarity search."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProductIndex:
    """FAISS-based product index for fast similarity search."""

    def __init__(self, dimension: int = 256):
        """Initialize ProductIndex with embedding dimension."""
        self.dimension = dimension
        self.index = None
        self.sku_ids = []

    def build(self, embeddings: np.ndarray, sku_ids: List[int]):
        """Build FAISS index from product embeddings."""
        try:
            import faiss

            # Store SKU IDs for result mapping
            self.sku_ids = sku_ids

            # Convert to float32 for FAISS
            embeddings = embeddings.astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)

            # Create inner product index (cosine after normalization)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Add all embeddings to index
            self.index.add(embeddings)

            logger.info(
                "FAISS index built: %d products, dim=%d", len(sku_ids), self.dimension
            )

        except ImportError:
            logger.warning("FAISS not installed. Run: pip install faiss-cpu")
            self.index = None

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """Search for similar products using query embedding."""
        # Return empty if index not built
        if self.index is None:
            return []

        # Reshape to 2D if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Convert to float32
        query_embedding = query_embedding.astype(np.float32)

        # Normalize query embedding
        try:
            import faiss

            faiss.normalize_L2(query_embedding)
        except ImportError:
            pass

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results list (filter invalid indices)
        results = [
            (self.sku_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

        return results

    def save(self, path: str):
        """Save index and metadata to file."""
        # Skip if index not built
        if self.index is not None:
            import faiss
            import json

            # Write FAISS index to disk
            faiss.write_index(self.index, path)

            # Prepare metadata
            metadata = {
                "sku_ids": self.sku_ids,
                "dimension": self.dimension,
            }
            
            # Write metadata JSON
            metadata_path = f"{path}.metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            logger.info("Index saved to %s (with metadata)", path)
