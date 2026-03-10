"""
Stage 2 Training: Query Encoder with Knowledge Distillation.

FAISS-based evaluation for apple-to-apple comparison with production.

Improvements over V2:
- Uses FAISS index search instead of np.dot() for MRR calculation
- Training MRR now matches production MRR
- All other training logic remains the same

Usage:
    python training/train_stage2.py --epochs 10 --batch_size 32
"""

import os
import sys
import platform
import argparse
import time
from datetime import datetime
import json
from typing import Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import faiss  # NEW: For FAISS-based evaluation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_query_encoder import EnhancedQueryEncoder
from models.search_model import SearchModel
from training.losses import DistillationLoss, HybridDistillationLoss
from training.augmenters import CharacterNoiseAugmenter
from training.utils.warmup_scheduler import WarmupCosineScheduler
from training.utils.data_alignment import DataAligner, load_embedding_metadata
from training.loader import load_products


class EnhancedDistillationDataset(Dataset):
    """Dataset with character noise and semantic query augmentation."""

    def __init__(
        self,
        product_names: List[str],
        target_embeddings: np.ndarray,
        tokenizer,
        augmenter: Optional[CharacterNoiseAugmenter] = None,
        semantic_augmenter=None,
        max_length: int = 32,
    ):
        assert len(product_names) == len(target_embeddings)
        self.product_names = product_names
        self.target_embeddings = target_embeddings
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.semantic_augmenter = semantic_augmenter
        self.max_length = max_length

    def __len__(self):
        return len(self.product_names)

    def __getitem__(self, idx):
        name = self.product_names[idx]
        target = self.target_embeddings[idx]

        # Apply semantic augmentation (word dropout, shuffle)
        if self.semantic_augmenter is not None:
            variants = self.semantic_augmenter.augment(name, n_augments=1)
            name = variants[0] if variants else name

        # Apply character noise (typos) to ALL queries including conceptual ones
        # With oversampling, each copy gets different noise = diverse training signal
        if self.augmenter is not None:
            name = self.augmenter.augment(name)

        # Tokenize augmented text
        encoded = self.tokenizer(
            name,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_embedding": torch.tensor(target, dtype=torch.float32),
        }


def load_product_names_for_training(
    checkpoint_path: str, start_date: str, end_date: str, cache_path: str = None
) -> Tuple[List[str], np.ndarray]:
    """Load and align products with embeddings, encoding new products before DataAligner.

    FIX for hybrid mode bug: DataAligner drops products without embeddings.
    New products are now encoded using Stage 1 model BEFORE DataAligner filtering.

    Returns:
        Tuple of (product_names, aligned_embeddings)
    """
    # Load embeddings from checkpoint
    emb_path = os.path.join(checkpoint_path, "product_embeddings.npy")
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings: shape={embeddings.shape}")

    # Try ClickHouse FIRST (primary source)
    print("[*] Loading products from ClickHouse (primary source)...")
    if cache_path:
        print(f"[*] Parquet cache override: {cache_path}")
    try:
        metadata = load_embedding_metadata(checkpoint_path)
        products_df = load_products(start_date, end_date, cache_path=cache_path)

        # === FIX: Encode NEW products before DataAligner ===
        existing_skus = set(int(sku) for sku in metadata["sku_order"])
        all_skus = set(products_df["sku_id"].astype(int).tolist())
        new_skus = all_skus - existing_skus

        print(f"[*] Existing products in checkpoint: {len(existing_skus)}")
        print(f"[*] New products (not in checkpoint): {len(new_skus)}")

        if new_skus:
            print(f"[*] Encoding {len(new_skus)} new products with Stage 1 model...")
            new_products_df = products_df[products_df["sku_id"].isin(new_skus)].copy()

            # Load Stage 1 model for encoding
            model_path = os.path.join(checkpoint_path, "best_model.pt")
            vocab_path = os.path.join(checkpoint_path, "category_vocab.json")

            with open(vocab_path, "r") as f:
                category_vocab = json.load(f)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vocab_sizes = {cat: len(v) for cat, v in category_vocab.items()}

            search_model = SearchModel(
                category_vocab_sizes=vocab_sizes, use_dora=True, embedding_dim=256
            )

            # Load model weights
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            search_model.load_state_dict(state_dict, strict=False)
            search_model.to(device)
            search_model.eval()
            search_model.set_category_vocab(category_vocab)

            # Encode new products
            category_values = {
                cat: new_products_df[cat].fillna("UNKNOWN").tolist()
                for cat in [
                    "division_name",
                    "dept_name",
                    "class_name",
                    "subclass_name",
                    "group_name",
                ]
            }

            with torch.no_grad():
                new_embeddings = search_model.encode_products(
                    product_names=new_products_df["sku_name"].fillna("").tolist(),
                    category_values=category_values,
                    batch_size=32,
                    device=device,
                    show_progress=True,
                )

            # Append new embeddings
            embeddings = np.vstack([embeddings, new_embeddings])

            # Update metadata with new SKUs (preserving order: existing first, then new)
            new_sku_order = list(metadata["sku_order"]) + [str(sku) for sku in new_skus]
            metadata["sku_order"] = new_sku_order

            print(f"[*] Updated embeddings shape: {embeddings.shape}")
            print(f"[*] Total SKUs in metadata: {len(new_sku_order)}")

        # NOW call DataAligner (all products have embeddings)
        aligner = DataAligner()
        aligned_names, aligned_embeddings = aligner.align_by_sku(
            products_df, embeddings, metadata
        )
        print(f"  Loaded {len(aligned_names)} products from ClickHouse")
        return aligned_names, aligned_embeddings

    except Exception as e:
        print(f"[!] ClickHouse failed: {e}")
        print("[*] Falling back to cache (safety net)...")

        # Fallback to cache (safety net)
        cache_path = os.path.join(checkpoint_path, "product_names_cache.json")
        if os.path.exists(cache_path):
            print(f"[*] Loading product names from cache: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            sku_names = cache_data["sku_names"]
            print(f"  Loaded {len(sku_names)} products from cache")
            return sku_names, embeddings
        else:
            raise RuntimeError(
                f"Cannot load products: ClickHouse failed and cache unavailable.\n"
                f"  ClickHouse error: {e}\n"
                f"  Cache path: {cache_path}\n\n"
                f"Cache is auto-generated by Stage 1. If missing, re-run Stage 1 first."
            )


def load_synthetic_queries(
    synthetic_paths, checkpoint_path: str, max_samples: int = None
) -> Tuple[List[str], np.ndarray]:
    """
    Load synthetic query-product pairs from one or more files and align with embeddings.

    Supports TWO formats:
    1. Legacy format (synthetic_queries.json): pairs with sku_id -> looks up embedding
    2. Clean format (synthetic_queries_clean.json): centroid_pairs with pre-computed
       centroid embeddings + individual_pairs with sku_id

    Args:
        synthetic_paths: Single path (str) or list of paths to synthetic JSON files.
        checkpoint_path: Path to Stage 1 checkpoint for embedding alignment.
        max_samples: Optional limit on total pairs loaded.

    Returns: (synthetic_queries, aligned_embeddings)
    """
    # Normalize to list
    if isinstance(synthetic_paths, str):
        synthetic_paths = [synthetic_paths]

    synthetic_queries = []
    aligned_embeddings = []

    # Load embeddings metadata (needed for both formats)
    metadata = load_embedding_metadata(checkpoint_path)
    sku_to_idx = {int(sku): idx for idx, sku in enumerate(metadata["sku_order"])}
    emb_path = os.path.join(checkpoint_path, "product_embeddings.npy")
    all_embeddings = np.load(emb_path)

    for path in synthetic_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Detect format: clean format has "centroid_pairs" key
        if "centroid_pairs" in data:
            # === CLEAN FORMAT (centroid + individual pairs) ===
            print(f"[*] Loading CLEAN synthetic data from {os.path.basename(path)}")

            # Load centroid pairs
            centroids = data.get("centroids", {})
            centroid_pairs = data.get("centroid_pairs", [])
            individual_pairs = data.get("individual_pairs", [])

            centroid_loaded = 0
            for pair in centroid_pairs:
                concept = pair["concept"]
                query = pair["query"]
                if concept in centroids:
                    emb = np.array(centroids[concept]["embedding"], dtype=np.float32)
                    synthetic_queries.append(query)
                    aligned_embeddings.append(emb)
                    centroid_loaded += 1

            individual_loaded = 0
            individual_skipped = 0
            for pair in individual_pairs:
                query = pair["query"]
                try:
                    sku_id = int(pair["sku_id"])
                except (ValueError, TypeError):
                    individual_skipped += 1
                    continue
                if sku_id in sku_to_idx:
                    synthetic_queries.append(query)
                    aligned_embeddings.append(all_embeddings[sku_to_idx[sku_id]])
                    individual_loaded += 1
                else:
                    individual_skipped += 1

            print(f"    Centroid pairs loaded: {centroid_loaded}")
            print(f"    Individual pairs loaded: {individual_loaded}")
            if individual_skipped > 0:
                print(f"    Individual pairs skipped: {individual_skipped}")

        else:
            # === LEGACY FORMAT (pairs with sku_id) ===
            print(f"[*] Loading LEGACY synthetic data from {os.path.basename(path)}")
            raw_pairs = data["pairs"]
            # Filter out hard negatives
            positive_pairs = [
                p
                for p in raw_pairs
                if p.get("is_positive", True) is not False
                and p.get("type", "") != "hard_negative"
            ]
            filtered_count = len(raw_pairs) - len(positive_pairs)
            print(f"    Loaded {len(positive_pairs)} positive pairs")
            if filtered_count > 0:
                print(f"    Filtered out {filtered_count} hard negative pairs")

            skipped = 0
            for pair in positive_pairs:
                try:
                    sku_id = int(pair["sku_id"])
                except (ValueError, TypeError):
                    skipped += 1
                    continue
                query = pair["query"]
                if sku_id in sku_to_idx:
                    synthetic_queries.append(query)
                    aligned_embeddings.append(all_embeddings[sku_to_idx[sku_id]])
            if skipped > 0:
                print(f"    Skipped {skipped} pairs with invalid SKU IDs")

    # Apply max_samples limit
    if max_samples and len(synthetic_queries) > max_samples:
        indices = list(range(len(synthetic_queries)))
        import random as _rng

        _rng.shuffle(indices)
        indices = indices[:max_samples]
        synthetic_queries = [synthetic_queries[i] for i in indices]
        aligned_embeddings = [aligned_embeddings[i] for i in indices]

    embeddings_array = np.array(aligned_embeddings, dtype=np.float32)
    print(f"[*] Total synthetic queries loaded: {len(synthetic_queries)}")

    return synthetic_queries, embeddings_array


def train_epoch(
    model: EnhancedQueryEncoder,
    dataloader: DataLoader,
    loss_fn: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler],
) -> float:
    """Train one epoch with distillation loss and mixed precision."""
    model.train()
    total_loss = 0.0

    # Track loss components
    total_alignment = 0.0
    total_uniformity = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target_embedding"].to(device)

        # Forward pass with mixed precision
        optimizer.zero_grad()

        # Forward pass with mixed precision (conditional)
        amp_enabled = scaler is not None
        with autocast("cuda", enabled=amp_enabled):
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, targets)

            # NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] Loss is {loss.item()}, skipping batch")
                optimizer.zero_grad()
                continue

            # Track loss components if available
            if hasattr(loss_fn, "get_loss_components"):
                components = loss_fn.get_loss_components(predictions, targets)
                total_alignment += components.get("alignment", 0)
                total_uniformity += components.get("uniformity", 0)
                num_batches += 1

        # Backward pass with scaler (conditional for CPU)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        # Update progress bar
        postfix = {
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if num_batches > 0:
            postfix["align"] = f"{total_alignment / num_batches:.3f}"
            postfix["unif"] = f"{total_uniformity / num_batches:.3f}"
        pbar.set_postfix(postfix)

    avg_loss = total_loss / len(dataloader)

    # Print loss component summary
    if num_batches > 0:
        print(
            f"  -> Alignment: {total_alignment / num_batches:.4f}, Uniformity: {total_uniformity / num_batches:.4f}"
        )

    return avg_loss


def validate(
    model: EnhancedQueryEncoder,
    dataloader: DataLoader,
    loss_fn: DistillationLoss,
    device: torch.device,
) -> dict:
    """Validate model and return loss and cosine similarity metrics."""
    model.eval()
    total_loss = 0.0
    all_cosine_sims = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_embedding"].to(device)

            # Forward pass
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Compute cosine similarity
            cos_sim = torch.cosine_similarity(predictions, targets, dim=1)
            all_cosine_sims.extend(cos_sim.cpu().tolist())

    return {
        "loss": total_loss / len(dataloader),
        "cosine_similarity": np.mean(all_cosine_sims),
        "cosine_std": np.std(all_cosine_sims),
    }


def evaluate_with_faiss(
    model: EnhancedQueryEncoder,
    dataloader: DataLoader,
    faiss_index: faiss.IndexFlatIP,
    device: torch.device,
    sample_size: int = 1000,
    index_offset: int = 0,
) -> dict:
    """FAISS-based evaluation matching production search behavior."""
    model.eval()
    all_predictions = []
    all_targets = []

    # Encode all validation queries
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="    Encoding queries", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_embedding"].to(device)

            predictions = model(input_ids, attention_mask)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.vstack(all_predictions).astype(np.float32)
    targets = np.vstack(all_targets).astype(np.float32)
    n_samples = len(predictions)

    metrics = {}

    # Sample for efficiency
    np.random.seed(42)
    sample_indices = np.random.choice(
        n_samples, min(sample_size, n_samples), replace=False
    )

    # FAISS search for MRR and Recall
    k = 20
    reciprocal_ranks = []
    recall_at_5 = 0
    recall_at_10 = 0

    print("    Running FAISS search for MRR calculation...")
    for idx in tqdm(sample_indices, desc="    FAISS eval", leave=False):
        query_emb = predictions[idx].reshape(1, -1)

        # Search FAISS index
        distances, indices = faiss_index.search(query_emb, k)

        # Check if correct index is in top-K (add offset for global index)
        global_idx = idx + index_offset
        retrieved_indices = indices[0].tolist()

        # Compute MRR: find rank of correct item
        if global_idx in retrieved_indices:
            rank = retrieved_indices.index(global_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank <= 5:
                recall_at_5 += 1
            if rank <= 10:
                recall_at_10 += 1
        else:
            reciprocal_ranks.append(0.0)

    n_evaluated = len(sample_indices)
    metrics["mrr"] = np.mean(reciprocal_ranks)
    metrics["recall@5"] = recall_at_5 / n_evaluated
    metrics["recall@10"] = recall_at_10 / n_evaluated

    # Compute anisotropy (average pairwise cosine)
    pred_norm = predictions / (
        np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8
    )
    sample_for_aniso = min(500, n_samples)
    sample_preds = pred_norm[
        np.random.choice(n_samples, sample_for_aniso, replace=False)
    ]
    sim_matrix = np.dot(sample_preds, sample_preds.T)
    mask = ~np.eye(sample_for_aniso, dtype=bool)
    metrics["anisotropy"] = np.mean(sim_matrix[mask])

    # Compute cosine similarity with targets
    target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(pred_norm * target_norm, axis=1)
    metrics["cosine_similarity"] = np.mean(cos_sims)

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced Query Encoder Training")
    parser.add_argument("--epochs", type=int, default=20)  # Default for full training
    parser.add_argument("--batch_size", type=int, default=64)  # Trial 14 best
    parser.add_argument("--lr", type=float, default=4.77e-05)  # Trial 14 best
    parser.add_argument(
        "--checkpoint_path", type=str, default="runtime/checkpoints/production_stage1"
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="Start epoch number (useful for resume)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to resume student model weights from",
    )
    parser.add_argument(
        "--output_path", type=str, default="runtime/checkpoints/query_encoder"
    )
    parser.add_argument(
        "--auto_timestamp", action="store_true", help="Add timestamp to output_path"
    )
    parser.add_argument("--start_date", type=str, default="2025-01-30")  # Full year
    parser.add_argument("--end_date", type=str, default="2026-01-30")  # Full year
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to products parquet cache (runtime/data/cache/products_catalog.parquet). "
        "If set and ClickHouse is unavailable, loads from parquet instead.",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.05,  # Trial 14 best
        help="Character noise probability (5%%)",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0529)  # Trial 14 best
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="hybrid",  # Trial 14 best
        choices=["cosine", "contrastive", "hybrid"],
        help="Loss function type: cosine, contrastive, or hybrid (recommended)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1247,  # Trial 14 best
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for alignment loss"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Weight for contrastive loss"
    )
    # V31: Configurable hybrid loss weights (Trial 14 best)
    parser.add_argument(
        "--alignment_weight",
        type=float,
        default=1.0,  # Trial 14 best
        help="HybridLoss alignment weight",
    )
    parser.add_argument(
        "--ranking_weight",
        type=float,
        default=0.25,  # v5.3: reduced from 0.4123 (Trial 14) for better generalization
        help="HybridLoss ranking weight",
    )
    parser.add_argument(
        "--uniformity_weight",
        type=float,
        default=0.1001,  # Trial 14 best
        help="HybridLoss uniformity weight",
    )
    parser.add_argument(
        "--hard_negative_weight",
        type=float,
        default=0.1521,  # Trial 14 best
        help="HybridLoss hard negative weight",
    )
    # Semantic query augmentation (enabled by default for better generalization)
    parser.add_argument(
        "--no_semantic_augment",
        action="store_true",
        help="Disable semantic query augmentation (word dropout, shuffle, truncation)",
    )
    # FIX: Max length should match Stage 1 (64 tokens)
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,  # Trial 14 best (matches Stage 1)
        help="Max token length for tokenizer (default: 64 to match Stage 1)",
    )
    # NEW: Synthetic conceptual query data for training (DEFAULT: enabled)
    parser.add_argument(
        "--no_synthetic_data",
        action="store_true",
        help="Disable synthetic conceptual queries (enabled by default)",
    )
    parser.add_argument(
        "--synthetic_data_path",
        type=str,
        nargs="+",
        default=["runtime/data/synthetic_merged.json"],
        help="Path(s) to synthetic queries JSON file(s). Supports multiple files.",
    )
    parser.add_argument(
        "--synthetic_ratio",
        type=float,
        default=0.08,  # v5.3: reduced from 0.3 to prevent synthetic bias (Strategy A)
        help="Ratio of synthetic to original samples (0.08 = 8%% of product count as synthetic, oversampled with augmentation)",
    )
    args = parser.parse_args()

    if args.auto_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_path = f"{args.output_path}_{timestamp}"
        print(f"Timestamped output path: {args.output_path}")

    # Print configuration
    print("\n[*] Training Configuration:")
    print("=" * 50)
    print(f"  Loss type: {args.loss_type}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Beta: {args.beta}")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and align data
    print("\n" + "=" * 60)
    print("LOADING AND ALIGNING DATA")
    print("=" * 60)

    from sklearn.model_selection import train_test_split

    product_names, target_embeddings = load_product_names_for_training(
        args.checkpoint_path,
        args.start_date,
        args.end_date,
        cache_path=args.cache_path,
    )
    print(f"Aligned {len(product_names)} products with embeddings")

    all_names = list(product_names)
    all_embeddings = list(target_embeddings)

    # Load synthetic data if enabled (default: enabled, use --no_synthetic_data to disable)
    if not args.no_synthetic_data:
        print("\n[*] Loading synthetic training data...")
        synthetic_names, synthetic_embeddings = load_synthetic_queries(
            synthetic_paths=args.synthetic_data_path,
            checkpoint_path=args.checkpoint_path,
        )

        # Calculate how many synthetic samples to add
        n_synthetic = int(len(product_names) * args.synthetic_ratio)

        # Shuffle before truncation to fix sequential bias
        # Current [:n_synthetic] takes from file top = 51% generic "belanja"/"kebutuhan"
        # Shuffle ensures representative sampling across all concepts
        import random

        combined = list(zip(synthetic_names, synthetic_embeddings))
        random.shuffle(combined)
        synthetic_names, synthetic_embeddings = zip(*combined) if combined else ([], [])
        synthetic_names = list(synthetic_names)
        synthetic_embeddings = list(synthetic_embeddings)

        # OVERSAMPLING: Repeat synthetic data to reach target ratio
        # Without this, 601 pairs in 70K total = 0.86% -- model barely sees conceptual queries
        # With oversampling, each copy gets DIFFERENT augmentation (char noise + semantic aug)
        # creating diverse training signal from the same base pairs
        n_original = len(synthetic_names)
        if n_original > 0 and n_original < n_synthetic:
            repeat_factor = (n_synthetic // n_original) + 1
            synthetic_names = (synthetic_names * repeat_factor)[:n_synthetic]
            synthetic_embeddings = (synthetic_embeddings * repeat_factor)[:n_synthetic]
            print(
                f"[*] Oversampled synthetic: {n_original} -> {len(synthetic_names)} "
                f"({repeat_factor}x repeat, each gets unique augmentation)"
            )
        elif n_original > n_synthetic:
            synthetic_names = synthetic_names[:n_synthetic]
            synthetic_embeddings = synthetic_embeddings[:n_synthetic]

        all_names.extend(synthetic_names)
        all_embeddings.extend(synthetic_embeddings)

        synthetic_pct = len(synthetic_names) / len(all_names) * 100
        print(
            f"[*] Added {len(synthetic_names)} synthetic samples ({synthetic_pct:.1f}% of total)"
        )
        print(f"[*] Total training samples: {len(all_names)}")

    # Split data into train/val
    print("\nSplitting data into train/val (80/20)...")
    train_names, val_names, train_embeddings_raw, val_embeddings_raw = train_test_split(
        all_names, all_embeddings, test_size=0.2, random_state=args.seed
    )

    train_embeddings = (
        np.array(train_embeddings_raw)
        if isinstance(train_embeddings_raw, list)
        else train_embeddings_raw
    )
    val_embeddings = (
        np.array(val_embeddings_raw)
        if isinstance(val_embeddings_raw, list)
        else val_embeddings_raw
    )

    # Data validation: check embeddings are valid
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    validation_passed = True

    # Check for NaN/Inf in embeddings
    if np.any(np.isnan(train_embeddings)):
        print("[!] VALIDATION FAILED: NaN found in train embeddings")
        validation_passed = False
    if np.any(np.isinf(train_embeddings)):
        print("[!] VALIDATION FAILED: Inf found in train embeddings")
        validation_passed = False
    if np.any(np.isnan(val_embeddings)):
        print("[!] VALIDATION FAILED: NaN found in val embeddings")
        validation_passed = False
    if np.any(np.isinf(val_embeddings)):
        print("[!] VALIDATION FAILED: Inf found in val embeddings")
        validation_passed = False

    # Check embedding shapes
    if train_embeddings.shape[1] != 256:
        print(
            f"[!] VALIDATION FAILED: Expected embedding dim 256, got {train_embeddings.shape[1]}"
        )
        validation_passed = False
    if val_embeddings.shape[1] != 256:
        print(
            f"[!] VALIDATION FAILED: Expected embedding dim 256, got {val_embeddings.shape[1]}"
        )
        validation_passed = False

    # Check train/val split ratio
    total_samples = len(train_names) + len(val_names)
    train_ratio = len(train_names) / total_samples
    val_ratio = len(val_names) / total_samples
    if not (0.75 <= train_ratio <= 0.85):
        print(
            f"[!] VALIDATION WARNING: Train ratio {train_ratio:.2f} outside expected [0.75, 0.85]"
        )
    if not (0.15 <= val_ratio <= 0.25):
        print(
            f"[!] VALIDATION WARNING: Val ratio {val_ratio:.2f} outside expected [0.15, 0.25]"
        )

    # Check names and embeddings match
    if len(train_names) != len(train_embeddings):
        print(
            f"[!] VALIDATION FAILED: Train names ({len(train_names)}) != embeddings ({len(train_embeddings)})"
        )
        validation_passed = False
    if len(val_names) != len(val_embeddings):
        print(
            f"[!] VALIDATION FAILED: Val names ({len(val_names)}) != embeddings ({len(val_embeddings)})"
        )
        validation_passed = False

    if validation_passed:
        print("[OK] All data validation checks PASSED")
    else:
        print("[!] VALIDATION FAILED - training may fail or produce poor results")
        print("    Please check your data and try again")
        return

    # Calculate train similarity stats
    n_sample = min(1000, len(train_embeddings))
    train_tensor = torch.from_numpy(train_embeddings[:n_sample]).float()
    sample_sims = torch.mm(train_tensor, train_tensor.T)
    mean_sim = (sample_sims.sum() - sample_sims.trace()) / (n_sample * (n_sample - 1))
    print(f"  Train: mean pairwise sim = {mean_sim.item():.4f}")

    # Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)

    model = EnhancedQueryEncoder(
        model_name="indobenchmark/indobert-lite-base-p1",  # IndoBERT-Lite for speed
        output_dim=256,
        freeze_base=False,
    ).to(device)

    # Resume from checkpoint if specified OR if start_epoch > 1
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.start_epoch > 1:
        # Auto-resume from output_path when continuing training
        resume_path = args.output_path
        print("\n[*] start_epoch > 1: Auto-resuming from output_path")

    if resume_path:
        checkpoint_file = os.path.join(resume_path, "enhanced_query_encoder.pt")
        if os.path.exists(checkpoint_file):
            print(f"\n[OK] Loading model from: {checkpoint_file}")
            checkpoint = torch.load(
                checkpoint_file, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print("[OK] Model weights loaded successfully!")

            # Restore scaler state if AMP was enabled
            if (
                checkpoint.get("amp_enabled", False)
                and "scaler_state_dict" in checkpoint
            ):
                resumed_scaler = GradScaler("cuda")
                resumed_scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print(
                    f"[OK] AMP scaler state restored! (scale={resumed_scaler.get_scale():.2e})"
                )
            else:
                resumed_scaler = None
                print("[*] No scaler state in checkpoint, will use fresh scaler")

        else:
            print(f"\n[!] Checkpoint not found: {checkpoint_file}")
            print("   Starting from scratch instead.")

    # Create augmenter and datasets
    augmenter = CharacterNoiseAugmenter(noise_prob=args.noise_prob)

    # Semantic augmenter (enabled by default for better generalization)
    semantic_aug = None
    if not args.no_semantic_augment:
        from training.augmenters import SemanticQueryAugmenter

        semantic_aug = SemanticQueryAugmenter(seed=args.seed)
        print("[*] Semantic Query Augmentation ENABLED")
    else:
        print("[*] Semantic Query Augmentation DISABLED (--no_semantic_augment)")

    # Create separate datasets for train and val
    # FIX: Use args.max_length instead of hardcoded 32
    train_dataset = EnhancedDistillationDataset(
        product_names=train_names,
        target_embeddings=train_embeddings,
        tokenizer=model.tokenizer,
        augmenter=augmenter,
        semantic_augmenter=semantic_aug,  # NEW
        max_length=args.max_length,
    )

    val_dataset = EnhancedDistillationDataset(
        product_names=val_names,
        target_embeddings=val_embeddings,
        tokenizer=model.tokenizer,
        augmenter=None,  # No augmentation for clean validation
        max_length=args.max_length,
    )

    # NEW: Noisy validation dataset (same augmenter as training) for dual evaluation
    val_dataset_noisy = EnhancedDistillationDataset(
        product_names=val_names,
        target_embeddings=val_embeddings,
        tokenizer=model.tokenizer,
        augmenter=augmenter,  # Same noise as training
        max_length=args.max_length,
    )

    # Note: No need for random_split since we already split data

    # Windows multiprocessing fix: fewer workers + no pin_memory
    is_windows = platform.system() == "Windows"
    num_workers = 2 if is_windows else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not is_windows,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not is_windows,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader_noisy = DataLoader(
        val_dataset_noisy,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not is_windows,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(
        f"Train: {len(train_dataset)}, Val (clean): {len(val_dataset)}, Val (noisy): {len(val_dataset_noisy)}"
    )

    # === BUILD FAISS INDEX FOR EVALUATION ===
    # This makes training evaluation apple-to-apple with production
    print("\n[*] Building FAISS index for evaluation...")
    all_embeddings = np.vstack([train_embeddings, val_embeddings]).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(all_embeddings.shape[1])
    faiss_index.add(all_embeddings)
    # We need to offset val indices since they come after train in the index
    val_index_offset = len(train_embeddings)
    print(f"   FAISS index built: {faiss_index.ntotal} products")

    # Initialize loss function based on type
    if args.loss_type == "hybrid":
        print(
            f"\n[*] Using HybridDistillationLoss (align={args.alignment_weight}, rank={args.ranking_weight}, unif={args.uniformity_weight}, hn={args.hard_negative_weight})"
        )
        loss_fn = HybridDistillationLoss(
            alignment_weight=args.alignment_weight,
            ranking_weight=args.ranking_weight,
            uniformity_weight=args.uniformity_weight,
            hard_negative_weight=args.hard_negative_weight,
            temperature=args.temperature,
        )
    elif args.loss_type == "contrastive":
        print(
            f"\n[*] Using OptimizedContrastiveDistillationLoss (temperature={args.temperature}, alpha={args.alpha}, beta={args.beta})"
        )

        # OPTIMIZED: Better gradient balancing for RTX 3050
        class OptimizedContrastiveDistillationLoss(nn.Module):
            def __init__(self, temperature=0.04, alpha=0.8, beta=0.2):
                super().__init__()
                self.temperature = temperature
                self.alpha = alpha
                self.beta = beta

            def forward(self, student_embeddings, teacher_embeddings):
                # Alignment loss (primary focus)
                cosine_sim = F.cosine_similarity(
                    student_embeddings, teacher_embeddings, dim=1
                )
                alignment_loss = (1 - cosine_sim).mean()

                # Uniformity loss (properly scaled)
                batch_size = student_embeddings.size(0)
                student_norm = F.normalize(student_embeddings, p=2, dim=1)
                sim_matrix = torch.mm(student_norm, student_norm.t()) / self.temperature
                mask = torch.eye(
                    batch_size, device=student_embeddings.device, dtype=torch.bool
                )
                uniformity_loss = torch.logsumexp(
                    sim_matrix.masked_fill(mask, float("-inf")), dim=1
                ).mean()

                # CRITICAL: Proper scaling to balance gradients
                return self.alpha * alignment_loss + self.beta * uniformity_loss * 0.05

            def get_loss_components(self, student_embeddings, teacher_embeddings):
                cosine_sim = F.cosine_similarity(
                    student_embeddings, teacher_embeddings, dim=1
                )
                alignment = (1 - cosine_sim).mean().item()

                batch_size = student_embeddings.size(0)
                student_norm = F.normalize(student_embeddings, p=2, dim=1)
                sim_matrix = torch.mm(student_norm, student_norm.t()) / self.temperature
                mask = torch.eye(
                    batch_size, device=student_embeddings.device, dtype=torch.bool
                )
                uniformity = (
                    torch.logsumexp(sim_matrix.masked_fill(mask, float("-inf")), dim=1)
                    .mean()
                    .item()
                )

                return {"alignment": alignment, "uniformity": uniformity * 0.05}

        loss_fn = OptimizedContrastiveDistillationLoss(
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        print("\n[*] Using DistillationLoss (cosine, margin=0.0)")
        loss_fn = DistillationLoss(margin=0.0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Calculate steps for remaining epochs
    num_epochs = args.epochs - args.start_epoch + 1
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING WITH ENHANCED DISTILLATION")
    print("=" * 60)

    # Initialize best_mrr and training_history (aligned with Optuna optimization target)
    # CRITICAL: Load existing values when resuming to prevent checkpoint overwrite
    best_mrr = 0.0
    training_history = []

    if args.start_epoch > 1:
        history_path = os.path.join(args.output_path, "training_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    training_history = json.load(f)
                # Extract best_mrr from previous training (aligned with Optuna)
                if training_history:
                    best_mrr = max(
                        h.get("mrr_combined", 0) or 0 for h in training_history
                    )
                print(f"[*] Loaded training history: {len(training_history)} epochs")
                print(f"[*] Previous best_mrr: {best_mrr:.4f}")
                print(
                    f"   (Only new epochs with mrr_combined > {best_mrr:.4f} will be saved)"
                )
            except Exception as e:
                print(f"[!] Could not load training history: {e}")
                print("   Starting with best_mrr=0.0 (may cause overwrite!)")
        else:
            print(f"[!] No training_history.json found at {history_path}")
            print("   Starting with best_mrr=0.0")

    # Smart early stopping: minimum 10 epochs, dynamic patience
    min_epochs = 10  # Minimum epochs before early stopping can trigger
    base_patience = 5  # Base patience value
    patience = base_patience
    no_improve = 0
    recent_improvements = []  # Track recent improvements for dynamic patience

    # NOTE: LR reduction on plateau is handled by WarmupCosineScheduler

    # Mixed Precision Scaler with CPU fallback
    device_type = device.type if isinstance(device, torch.device) else device.type
    if device_type == "cuda":
        # Use restored scaler from checkpoint if available, otherwise create fresh
        if "resumed_scaler" in dir() and resumed_scaler is not None:
            scaler = resumed_scaler
            print(
                f"[*] Mixed Precision (AMP) ENABLED (resumed, scale={scaler.get_scale():.2e})"
            )
        else:
            scaler = GradScaler("cuda")
            print("[*] Mixed Precision (AMP) ENABLED (fresh scaler)")
    else:
        scaler = None
        print("[*] Mixed Precision (AMP) DISABLED (CPU mode)")

    # Track total training time
    training_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        start_time = time.time()

        # Memory monitoring for OOM prevention
        if device_type == "cuda":
            gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            gpu_mem_total = (
                torch.cuda.get_device_properties(device).total_memory / 1024**3
            )  # GB
            mem_usage_pct = (gpu_mem_allocated / gpu_mem_total) * 100

            # Warn if memory usage is high (>85%)
            if mem_usage_pct > 85:
                print(
                    f"  [!] HIGH GPU MEMORY: {gpu_mem_allocated:.2f}/{gpu_mem_total:.2f} GB ({mem_usage_pct:.1f}%)"
                )
                print(
                    f"      Risk of OOM! Consider reducing batch_size or clearing cache."
                )
            elif epoch == 1 or epoch % 5 == 0:
                # Log memory status periodically
                mem_msg = f"  [*] GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved ({mem_usage_pct:.1f}% used)"

                # Add scaler monitoring if AMP enabled
                if scaler is not None:
                    scale = scaler.get_scale()
                    mem_msg += f" | AMP Scale: {scale:.2e}"
                    # Warn if scale is very small (indicates gradient issues)
                    if scale < 1e-4:
                        mem_msg += " [!] Very low scale - gradient issues detected!"

                print(mem_msg)

        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device, epoch, scaler
        )
        val_metrics = validate(model, val_loader, loss_fn, device)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Cosine: {val_metrics['cosine_similarity']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Comprehensive evaluation every 5 epochs (or last epoch) - optimized for speed
        if epoch % 5 == 0 or epoch == args.epochs:
            print("\n    Running DUAL evaluation (clean + noisy) with FAISS...")

            # 1. Clean evaluation with FAISS
            comp_metrics_clean = evaluate_with_faiss(
                model,
                val_loader,
                faiss_index,
                device,
                sample_size=1000,
                index_offset=val_index_offset,
            )

            # 2. Noisy evaluation with FAISS
            comp_metrics_noisy = evaluate_with_faiss(
                model,
                val_loader_noisy,
                faiss_index,
                device,
                sample_size=1000,
                index_offset=val_index_offset,
            )

            # Combined MRR (same weights as Optuna: 30% clean, 70% noisy)
            WEIGHT_CLEAN = 0.3
            WEIGHT_NOISY = 0.7
            mrr_clean = comp_metrics_clean.get("mrr", 0)
            mrr_noisy = comp_metrics_noisy.get("mrr", 0)
            mrr_combined = WEIGHT_CLEAN * mrr_clean + WEIGHT_NOISY * mrr_noisy

            print(
                f"    [*] MRR Clean: {mrr_clean:.4f} | MRR Noisy: {mrr_noisy:.4f} | MRR Combined: {mrr_combined:.4f}"
            )
            print(
                f"    [*] Recall@5 Clean: {comp_metrics_clean.get('recall@5', 0):.4f} | Noisy: {comp_metrics_noisy.get('recall@5', 0):.4f}"
            )
            print(f"    [*] Anisotropy: {comp_metrics_clean.get('anisotropy', 0):.4f}")

            # Overfitting detection
            overfitting_gap = train_loss - val_metrics["loss"]
            if overfitting_gap > 0.3:
                print(
                    f"    [!] POTENTIAL OVERFITTING: train-val gap = {overfitting_gap:.4f}"
                )

            # Add comprehensive metrics to history
            val_metrics.update(
                {
                    "mrr_clean": mrr_clean,
                    "mrr_noisy": mrr_noisy,
                    "mrr_combined": mrr_combined,
                    "recall@5": comp_metrics_clean.get("recall@5"),
                    "recall@5_noisy": comp_metrics_noisy.get("recall@5"),
                    "anisotropy": comp_metrics_clean.get("anisotropy"),
                    "overfitting_gap": overfitting_gap,
                }
            )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_cosine": val_metrics["cosine_similarity"],
                "mrr_clean": val_metrics.get("mrr_clean"),
                "mrr_noisy": val_metrics.get("mrr_noisy"),
                "mrr_combined": val_metrics.get("mrr_combined"),
                "recall@5": val_metrics.get("recall@5"),
                "recall@5_noisy": val_metrics.get("recall@5_noisy"),
                "anisotropy": val_metrics.get("anisotropy"),
                "overfitting_gap": val_metrics.get("overfitting_gap"),
            }
        )

        # Save best model based on MRR combined (aligned with Optuna optimization)
        # No quality gates - always save best model (like Stage 1)
        # Deployment decision is separate (handled by api_training.py)
        current_mrr = val_metrics.get("mrr_combined", 0) or 0

        if current_mrr > best_mrr:
            # Calculate improvement BEFORE updating best_mrr
            improvement = current_mrr - best_mrr
            best_mrr = current_mrr
            no_improve = 0
            model.save(args.output_path, scaler=scaler)
            print(
                f"  -> Best model saved! (mrr_combined: {best_mrr:.4f}, improvement: +{improvement:.4f})"
            )

            # Track improvement for dynamic patience
            recent_improvements.append(improvement)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)

            # Dynamic patience: increase if still improving consistently
            if len(recent_improvements) >= 3 and sum(recent_improvements[-3:]) > 0.01:
                patience = min(base_patience + 3, 10)  # Cap at 10
                if epoch % 5 == 0:
                    print(
                        f"  [*] Dynamic patience: {patience} (consistent improvements detected)"
                    )
        else:
            no_improve += 1

            # Smart early stopping: only trigger after minimum epochs
            if no_improve >= patience and epoch >= min_epochs:
                print(
                    f"\n[STOP] Early stopping at epoch {epoch} after {no_improve} epochs without improvement"
                )
                print(
                    f"       Minimum epochs ({min_epochs}) satisfied, patience: {patience}"
                )
                break
            elif no_improve >= patience and epoch < min_epochs:
                print(
                    f"  [*] Reached patience ({no_improve}/{patience}) but epoch {epoch} < min_epochs ({min_epochs})"
                )
                print(f"      Continuing training to ensure convergence...")

    # Save training history (convert numpy types for JSON compatibility)
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(args.output_path, exist_ok=True)
    with open(
        os.path.join(args.output_path, "training_history.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(convert_to_json_serializable(training_history), f, indent=2)

    # Save training metadata (unified format)
    best_metrics = {}
    if training_history:
        # Find entry with best mrr_combined
        best_entry = max(training_history, key=lambda x: x.get("mrr_combined", 0) or 0)
        best_metrics = best_entry

    metadata = {
        "created_at": datetime.now().isoformat(),
        "data_range": {"start_date": args.start_date, "end_date": args.end_date},
        "metrics": convert_to_json_serializable(best_metrics),
        "hyperparams": vars(args),
    }

    with open(os.path.join(args.output_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("  [OK] Saved training_metadata.json")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    # Calculate total training time
    total_time_minutes = (time.time() - training_start_time) / 60
    print(f"Total Training Time: {total_time_minutes:.1f} minutes")
    print(f"Best MRR combined: {best_mrr:.4f}")
    print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
