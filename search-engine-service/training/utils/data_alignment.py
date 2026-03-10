"""Data alignment for knowledge distillation using SKU IDs as primary key."""

import os
import logging
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np


class DataAlignmentError(Exception):
    """Custom exception for data alignment issues."""

    pass


class DataAligner:
    """Align product names with target embeddings using SKU IDs."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data alignment operations."""
        logger = logging.getLogger("DataAligner")
        logger.setLevel(logging.INFO)

        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def align_by_sku(
        self,
        products_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_metadata: Dict[str, Any],
    ) -> Tuple[List[str], np.ndarray]:
        """Align product names with target embeddings using SKU IDs."""
        self.logger.info("Starting SKU-based data alignment...")

        # Validate inputs
        self._validate_inputs(products_df, embeddings, embedding_metadata)

        # Get SKU order from metadata
        if "sku_order" not in embedding_metadata:
            raise DataAlignmentError(
                "embedding_metadata must contain 'sku_order' key with SKU ordering information"
            )

        embedding_sku_order = embedding_metadata["sku_order"]

        # Ensure SKU ID type consistency
        products_df, embedding_sku_order = self._normalize_sku_types(
            products_df, embedding_sku_order
        )

        # Validate counts before alignment
        self._validate_counts(products_df, embeddings, embedding_sku_order)

        # Perform alignment
        aligned_names, aligned_embeddings = self._perform_alignment(
            products_df, embeddings, embedding_sku_order
        )

        # Final validation
        self._validate_alignment_result(aligned_names, aligned_embeddings)

        self.logger.info(f"✅ Successfully aligned {len(aligned_names)} products")
        return aligned_names, aligned_embeddings

    def _validate_inputs(
        self,
        products_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_metadata: Dict[str, Any],
    ) -> None:
        """Validate input data types and structure."""
        # Validate DataFrame
        if not isinstance(products_df, pd.DataFrame):
            raise DataAlignmentError("products_df must be a pandas DataFrame")

        # Check required columns
        required_columns = ["sku_id", "sku_name"]
        missing_columns = [
            col for col in required_columns if col not in products_df.columns
        ]
        if missing_columns:
            raise DataAlignmentError(
                f"products_df missing required columns: {missing_columns}"
            )

        # Validate embeddings
        if not isinstance(embeddings, np.ndarray):
            raise DataAlignmentError("embeddings must be a numpy array")

        if embeddings.ndim != 2:
            raise DataAlignmentError(
                f"embeddings must be 2D array, got {embeddings.ndim}D"
            )

        # Validate metadata
        if not isinstance(embedding_metadata, dict):
            raise DataAlignmentError("embedding_metadata must be a dictionary")

    def _normalize_sku_types(
        self, products_df: pd.DataFrame, embedding_sku_order: List[Any]
    ) -> Tuple[pd.DataFrame, List[Any]]:
        """Ensure SKU ID types are consistent between products and embeddings."""
        products_df = products_df.copy()

        # Detect types
        product_sku_type = (
            type(products_df["sku_id"].iloc[0]) if len(products_df) > 0 else str
        )
        embedding_sku_type = (
            type(embedding_sku_order[0]) if len(embedding_sku_order) > 0 else str
        )

        self.logger.info(f"Product SKU type: {product_sku_type.__name__}")
        self.logger.info(f"Embedding SKU type: {embedding_sku_type.__name__}")

        # Convert to consistent type (prefer int, fallback to string)
        if product_sku_type != embedding_sku_type:
            self.logger.warning("SKU ID type mismatch detected, normalizing types")

            # Try int first, fallback to string
            try:
                products_df["sku_id"] = products_df["sku_id"].astype(int)
                embedding_sku_order = [int(sku) for sku in embedding_sku_order]
                self.logger.info("Normalized to int type")
            except (ValueError, TypeError):
                products_df["sku_id"] = products_df["sku_id"].astype(str)
                embedding_sku_order = [str(sku) for sku in embedding_sku_order]
                self.logger.info("Normalized to str type")

        return products_df, embedding_sku_order

    def _validate_counts(
        self,
        products_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_sku_order: List[Any],
    ) -> None:
        """Validate that counts are consistent across all data sources."""
        product_count = len(products_df)
        embedding_count = len(embeddings)
        sku_order_count = len(embedding_sku_order)

        self.logger.info(
            f"Data counts - Products: {product_count}, "
            f"Embeddings: {embedding_count}, SKU order: {sku_order_count}"
        )

        # Check embedding vs SKU order consistency
        if embedding_count != sku_order_count:
            raise DataAlignmentError(
                f"Embedding count ({embedding_count}) != SKU order count ({sku_order_count}). "
                "This indicates corrupted embedding metadata."
            )

        # Check for overlap
        product_skus = set(products_df["sku_id"].unique())
        embedding_skus = set(embedding_sku_order)

        overlap = product_skus.intersection(embedding_skus)
        overlap_ratio = len(overlap) / max(len(product_skus), len(embedding_skus))

        self.logger.info(
            f"SKU overlap: {len(overlap)}/{max(len(product_skus), len(embedding_skus))} "
            f"({overlap_ratio:.2%})"
        )

        # Warn if low overlap
        if overlap_ratio < 0.8:
            self.logger.warning(
                f"Low SKU overlap ({overlap_ratio:.2%}). "
                "This may indicate data quality issues."
            )

    def _perform_alignment(
        self,
        products_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_sku_order: List[Any],
    ) -> Tuple[List[str], np.ndarray]:
        """Perform the actual alignment based on SKU matching."""
        # Create SKU to embedding index mapping
        sku_to_idx = {sku: idx for idx, sku in enumerate(embedding_sku_order)}

        # Filter products to only those with embeddings
        products_with_embeddings = products_df[
            products_df["sku_id"].isin(sku_to_idx.keys())
        ].copy()

        # Check if any products have embeddings
        if len(products_with_embeddings) == 0:
            raise DataAlignmentError(
                "No products found with matching embeddings. "
                "Check SKU ID consistency and data sources."
            )

        # Sort products by embedding order for perfect alignment
        products_with_embeddings["embedding_idx"] = products_with_embeddings[
            "sku_id"
        ].map(sku_to_idx)
        products_with_embeddings = products_with_embeddings.sort_values("embedding_idx")

        # Extract aligned data
        aligned_names = products_with_embeddings["sku_name"].tolist()
        embedding_indices = products_with_embeddings["embedding_idx"].tolist()
        aligned_embeddings = embeddings[embedding_indices]

        # Log alignment statistics
        original_product_count = len(products_df)
        aligned_count = len(aligned_names)
        dropped_count = original_product_count - aligned_count

        if dropped_count > 0:
            self.logger.warning(f"Dropped {dropped_count} products without embeddings")

        return aligned_names, aligned_embeddings

    def _validate_alignment_result(
        self, aligned_names: List[str], aligned_embeddings: np.ndarray
    ) -> None:
        """Validate the final alignment result."""
        # Check count consistency
        if len(aligned_names) != len(aligned_embeddings):
            raise DataAlignmentError(
                f"Alignment failed: {len(aligned_names)} names != {len(aligned_embeddings)} embeddings"
            )

        # Check for empty names
        empty_names = [
            i for i, name in enumerate(aligned_names) if not name or not name.strip()
        ]
        if empty_names:
            self.logger.warning(
                f"Found {len(empty_names)} empty product names at indices: {empty_names[:10]}"
            )

        # Check embedding dimensions
        if aligned_embeddings.ndim != 2:
            raise DataAlignmentError(
                f"Aligned embeddings must be 2D, got {aligned_embeddings.ndim}D"
            )

        self.logger.info(
            f"Final alignment: {len(aligned_names)} products, "
            f"embedding shape: {aligned_embeddings.shape}"
        )

    def create_embedding_metadata(
        self, products_df: pd.DataFrame, output_path: str
    ) -> Dict[str, Any]:
        """Create embedding metadata for future alignment."""
        metadata = {
            "sku_order": products_df["sku_id"].tolist(),
            "product_count": len(products_df),
            "creation_timestamp": pd.Timestamp.now().isoformat(),
            "sku_id_type": (
                str(type(products_df["sku_id"].iloc[0]).__name__)
                if len(products_df) > 0
                else "str"
            ),
        }

        # Save metadata
        import json

        metadata_path = os.path.join(output_path, "embedding_metadata.json")
        os.makedirs(output_path, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved embedding metadata to {metadata_path}")
        return metadata


def load_embedding_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """Load embedding metadata from checkpoint directory."""
    import json

    metadata_path = os.path.join(checkpoint_path, "embedding_metadata.json")

    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Embedding metadata not found at {metadata_path}. "
            "This file is required for proper data alignment. "
            "Please regenerate embeddings with updated training script."
        )

    try:
        # Load metadata JSON
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Validate required fields
        required_fields = ["sku_order", "product_count"]
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise DataAlignmentError(
                f"Metadata missing required fields: {missing_fields}"
            )

        return metadata

    except json.JSONDecodeError as e:
        raise DataAlignmentError(f"Invalid metadata JSON: {e}")
    except Exception as e:
        raise DataAlignmentError(f"Failed to load metadata: {e}")
