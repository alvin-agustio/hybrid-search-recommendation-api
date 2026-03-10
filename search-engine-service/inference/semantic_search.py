"""Direct semantic search using query encoder and FAISS index."""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DirectSemanticSearcher:
    """Direct semantic search: Query → Encoder → FAISS."""

    def __init__(
        self,
        encoder,
        embeddings: np.ndarray,
        sku_ids: List[int],
        device: Optional[torch.device] = None,
    ):
        """Initialize semantic searcher with encoder and product embeddings."""
        # Store encoder and SKU mapping
        self.encoder = encoder
        self.sku_ids = sku_ids

        # Set device (GPU if available, else CPU)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move encoder to device and set to eval mode
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        # Store embeddings (already L2-normalized from model)
        self.normalized_embeddings = embeddings.astype(np.float32)

        # Build FAISS index for fast search
        self.index = None
        self._build_index()

    def _build_index(self):
        """Build FAISS index for fast similarity search."""
        try:
            import faiss

            # Get embedding dimension
            dim = self.normalized_embeddings.shape[1]

            # Create Inner Product index (cosine after normalization)
            self.index = faiss.IndexFlatIP(dim)

            # Add all product embeddings to index
            self.index.add(self.normalized_embeddings)

            logger.info(
                "FAISS index built: %d products, dim=%d", len(self.sku_ids), dim
            )
        except ImportError:
            # Fallback to numpy if FAISS not available
            logger.warning("FAISS not installed, using numpy fallback")
            self.index = None

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search for products similar to query."""
        # Encode query to embedding vector
        query_emb = self.encoder.encode(query, device=self.device)

        # Convert to numpy if torch tensor
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.cpu().numpy().astype(np.float32)
        else:
            query_emb = query_emb.astype(np.float32)

        # Reshape to 2D for FAISS (1, dim)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Search using FAISS or numpy fallback
        if self.index is not None:
            # FAISS search (fast)
            scores, indices = self.index.search(query_emb, top_k)

            # Build results (skip invalid indices)
            results = [
                (self.sku_ids[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        else:
            # Numpy fallback (slower)
            similarities = np.dot(self.normalized_embeddings, query_emb.T).flatten()

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Build results
            results = [
                (self.sku_ids[idx], float(similarities[idx])) for idx in top_indices
            ]

        # Return (sku_id, score) tuples
        return results
