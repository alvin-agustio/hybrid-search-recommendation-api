"""
Training Script - Stage 1 (Product Embeddings with Lexical Adversary).

UPGRADES:
- HybridSampler: 50% normal PK + 50% keyword conflict batches
- Text Augmentation: Random brand/size removal
- Collision Metric: Tracks "same word, wrong category" errors
"""

import os
import sys
import warnings

# Suppress known PyTorch warmup scheduler warning (not fatal)
warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step.*before.*optimizer.step")

import argparse
import time
from datetime import datetime
import json
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

os.environ["HF_HUB_DISABLE_VULNERABILITY_CHECK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.losses import InfoNCELoss

CAT_COLS = ["division_name", "dept_name", "class_name", "subclass_name", "group_name"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_keyword_conflicts(df, min_categories=3):
    """Build keyword conflict index: words appearing in 3+ different categories."""
    word_to_categories = defaultdict(set)
    word_to_skus = defaultdict(list)

    for idx, row in df.iterrows():
        words = set(str(row["sku_name"]).upper().split())
        subclass = row["subclass_name"]
        for word in words:
            if len(word) >= 3:  # Ignore short words
                word_to_categories[word].add(subclass)
                word_to_skus[word].append(idx)

    # Keep only conflict words (appear in 3+ categories)
    conflicts = {
        word: {"categories": list(cats), "skus": word_to_skus[word]}
        for word, cats in word_to_categories.items()
        if len(cats) >= min_categories
    }
    return conflicts


def augment_text(text, rng):
    """Simulate short user queries by randomly truncating text."""
    # 50% chance to augment for better query simulation
    if rng.random() < 0.5:
        words = text.split()
        if len(words) > 2:
            # Take only 2-3 first words (simulate short query)
            num_words = rng.randint(2, min(3, len(words)))
            text = " ".join(words[:num_words])
    return text if text.strip() else "UNKNOWN"


class HybridSampler(Sampler):
    """Hybrid PK Sampler: 50% normal PK batches + 50% keyword conflict batches for adversarial training."""

    def __init__(self, labels, df, m=4, p=8, seed=42):
        super().__init__()
        self.m = m
        self.p = p
        self.batch_size = m * p
        self.labels = np.array(labels)
        self.df = df

        # Normal PK structures
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        self.valid_labels = list(self.label_to_indices.keys())

        if len(self.valid_labels) < p:
            raise ValueError(f"Need {p} classes, only have {len(self.valid_labels)}")

        # Build keyword conflict index
        self.conflicts = build_keyword_conflicts(df)
        self.conflict_words = [
            w for w, d in self.conflicts.items() if len(d["skus"]) >= self.batch_size
        ]
        print(
            f"  Found {len(self.conflict_words)} conflict keywords for adversarial training"
        )

        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self._num_pk_batches = len(self.valid_labels) // p
        self._num_conflict_batches = min(len(self.conflict_words), self._num_pk_batches)
        self._total_batches = self._num_pk_batches + self._num_conflict_batches

    def __iter__(self):
        rng = np.random.default_rng(self.epoch + 42)

        # Generate PK batches
        pk_batches = []
        labels = self.valid_labels.copy()
        rng.shuffle(labels)
        for batch_idx in range(self._num_pk_batches):
            batch = []
            batch_labels = labels[batch_idx * self.p : (batch_idx + 1) * self.p]
            for label in batch_labels:
                indices = self.label_to_indices[label]
                replace = len(indices) < self.m
                selected = rng.choice(indices, self.m, replace=replace)
                batch.extend(selected.tolist())
            pk_batches.append(batch)

        # Generate Conflict batches
        conflict_batches = []
        if self.conflict_words:
            conflict_words = self.conflict_words.copy()
            rng.shuffle(conflict_words)
            for word in conflict_words[: self._num_conflict_batches]:
                skus = self.conflicts[word]["skus"]
                categories = self.conflicts[word]["categories"]
                # Enforce diversity: need at least 3 different categories
                if len(skus) >= self.batch_size and len(categories) >= 3:
                    # Sample ensuring category diversity
                    sku_to_cat = {}
                    for sku_idx in skus:
                        cat = self.df.iloc[sku_idx]["subclass_name"]
                        sku_to_cat[sku_idx] = cat

                    # Sample from each category to ensure diversity
                    selected = []
                    cats_used = set()
                    remaining_skus = list(skus)
                    rng.shuffle(remaining_skus)

                    for sku_idx in remaining_skus:
                        selected.append(sku_idx)
                        cats_used.add(sku_to_cat[sku_idx])
                        if len(selected) >= self.batch_size:
                            break

                    if len(cats_used) >= 3:  # Only add if diverse enough
                        conflict_batches.append(selected[: self.batch_size])

        # Interleave PK and Conflict batches
        all_batches = pk_batches + conflict_batches
        rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self._total_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def notify_dataset_epoch(self, dataset):
        """Notify dataset of current epoch for varying augmentation."""
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)


class PreTokenizedProductDataset(Dataset):
    """Pre-tokenized product dataset with epoch-varying text augmentation."""

    def __init__(
        self,
        df,
        tokenizer,
        category_vocab,
        max_length=64,
        augment=True,
        text_column="sku_name",
    ):
        self.df = df.reset_index(drop=True)
        self.category_vocab = category_vocab
        self.augment = augment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.epoch = 0
        self.text_column = text_column

        # Pre-tokenize all texts (base encodings without augmentation)
        print(f"  Pre-tokenizing products using '{text_column}'...")
        texts = df[text_column].tolist()
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.raw_texts = texts

        # Build category ID tensors
        self.category_ids = {}
        for col in CAT_COLS:
            self.category_ids[col] = torch.tensor(
                [
                    category_vocab.get(col, {}).get(row[col], 0)
                    for _, row in df.iterrows()
                ]
            )

        # Build subclass label tensor
        self.subclass_labels = torch.tensor(
            [
                category_vocab.get("subclass_name", {}).get(row["subclass_name"], 0)
                for _, row in df.iterrows()
            ]
        )

        print(f"  Pre-tokenized {len(df)} products")

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        # Apply augmentation with epoch-varying seed
        rng = random.Random(idx + self.epoch * 100000 + 42)
        if self.augment and rng.random() < 0.3:
            # Augment text and re-tokenize
            aug_text = augment_text(self.raw_texts[idx], rng)
            enc = self.tokenizer(
                aug_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            # Use pre-tokenized encodings
            input_ids = self.encodings["input_ids"][idx]
            attention_mask = self.encodings["attention_mask"][idx]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "category_ids": {col: self.category_ids[col][idx] for col in CAT_COLS},
            "subclass_label": self.subclass_labels[idx],
        }

        return result


def collate_products(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    subclass_labels = torch.stack([b["subclass_label"] for b in batch])
    category_ids = {}
    for col in CAT_COLS:
        category_ids[col] = torch.stack([b["category_ids"][col] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "category_ids": category_ids,
        "subclass_labels": subclass_labels,
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    triplet_weight=1.0,
    class_weight=0.3,
    gate_l1_weight=0.1,
    temperature=0.07,
    category_dropout=0.0,
):
    """Train one epoch with InfoNCE loss and optional category dropout."""
    model.train()
    total_loss, triplet_losses, class_losses = 0, 0, 0
    n_batches = 0
    gate_means = []

    # Initialize InfoNCE loss
    triplet_loss_fn = InfoNCELoss(temperature=temperature)

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        category_ids = {k: v.to(device) for k, v in batch["category_ids"].items()}
        subclass_labels = batch["subclass_labels"].to(device)

        # Apply category dropout (zero out embeddings randomly)
        if category_dropout > 0 and model.training:
            for col in category_ids:
                mask = (
                    torch.rand(category_ids[col].shape[0], device=device)
                    < category_dropout
                )
                category_ids[col] = category_ids[col].clone()
                category_ids[col][mask] = 0

        optimizer.zero_grad()

        # Forward pass with mixed precision
        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if model.classification_head is not None:
                embeddings, logits = model(
                    input_ids, attention_mask, category_ids, return_logits=True
                )
            else:
                embeddings = model(input_ids, attention_mask, category_ids)
                logits = None

            # Compute losses
            triplet_loss = triplet_loss_fn(embeddings, subclass_labels)
            class_loss = (
                F.cross_entropy(logits, subclass_labels)
                if logits is not None
                else torch.tensor(0.0, device=device)
            )

            # Gate L1 regularization
            gate_l1 = torch.tensor(0.0, device=device)
            if hasattr(model, "fusion") and model.fusion.last_gate is not None:
                gate_l1 = model.fusion.last_gate.mean()
                gate_means.append(gate_l1.item())

            # Combined loss
            loss = (
                triplet_weight * triplet_loss
                + class_weight * class_loss
                + gate_l1_weight * gate_l1
            )

        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Scheduler step AFTER optimizer step
        scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        triplet_losses += triplet_loss.item()
        class_losses += class_loss.item()
        n_batches += 1

        # Log progress
        if batch_idx % 20 == 0:
            print(
                f"  Batch {batch_idx}/{len(dataloader)}, Loss: {total_loss / n_batches:.4f}"
            )

    gate_mean = sum(gate_means) / len(gate_means) if gate_means else 0.5
    print(f"  Gate mean: {gate_mean:.4f}")
    return gate_mean


def validate_with_collision(model, val_df, category_vocab, device, tokenizer, k=10):
    """Validate model with NDCG, MRR, and collision rate metrics."""
    from inference.faiss_index import ProductIndex
    from training.utils.metrics import ndcg_at_k

    model.eval()
    embeddings = []

    # Generate embeddings for validation set
    with torch.no_grad():
        for start in range(0, len(val_df), 64):
            batch_df = val_df.iloc[start : start + 64]
            texts = batch_df["sku_name"].tolist()
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            category_ids = {
                col: torch.tensor(
                    [
                        category_vocab.get(col, {}).get(row[col], 0)
                        for _, row in batch_df.iterrows()
                    ]
                ).to(device)
                for col in CAT_COLS
            }

            emb = model(encoded["input_ids"], encoded["attention_mask"], category_ids)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    sku_list = val_df["sku_id"].tolist()
    sku_to_idx = {sku: idx for idx, sku in enumerate(sku_list)}
    sku_to_name = dict(zip(val_df["sku_id"], val_df["sku_name"]))

    # Build FAISS index
    index = ProductIndex(dimension=embeddings.shape[1])
    index.build(embeddings, sku_list)

    # Select categories with enough samples
    cat_counts = val_df["subclass_name"].value_counts()
    valid_cats = cat_counts[cat_counts >= 3].index.tolist()
    rng = random.Random(42)
    ndcg_scores, mrr_scores, collision_counts = [], [], []

    # Evaluate on 200 random queries
    for _ in range(min(200, len(valid_cats) * 2)):
        cat = rng.choice(valid_cats)
        cat_skus = val_df[val_df["subclass_name"] == cat]["sku_id"].tolist()
        if len(cat_skus) < 2:
            continue
        query_sku = rng.choice(cat_skus)
        relevant = set(cat_skus) - {query_sku}

        if query_sku not in sku_to_idx:
            continue
        query_emb = embeddings[sku_to_idx[query_sku]]
        query_name = sku_to_name.get(query_sku, "")
        query_words = set(query_name.upper().split())

        # Search and compute metrics
        results = index.search(query_emb, top_k=k + 1)
        retrieved = [r[0] for r in results if r[0] != query_sku][:k]

        ndcg_scores.append(ndcg_at_k(retrieved, relevant, k=k))

        # MRR: reciprocal rank of first relevant result
        mrr = 0.0
        for rank, r_sku in enumerate(retrieved):
            if r_sku in relevant:
                mrr = 1.0 / (rank + 1)
                break
        mrr_scores.append(mrr)

        # Collision: same word but different category
        collisions = 0
        for r_sku in retrieved:
            if r_sku not in relevant:
                r_name = sku_to_name.get(r_sku, "")
                r_words = set(r_name.upper().split())
                if query_words & r_words:
                    collisions += 1
        collision_counts.append(collisions)

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    avg_collision = np.mean(collision_counts) if collision_counts else 0.0
    print(
        f"  NDCG@{k}: {avg_ndcg:.4f} | MRR@{k}: {avg_mrr:.4f} | Collision: {avg_collision:.2f}/query"
    )
    return avg_ndcg, avg_collision


def create_scheduler(optimizer, num_steps, warmup_ratio=0.1):
    """Create cosine annealing scheduler with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup = int(num_steps * warmup_ratio)

    def lr_lambda(step):
        # Linear warmup
        if step < warmup:
            return step / max(1, warmup)
        # Cosine annealing
        progress = (step - warmup) / max(1, num_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def generate_embeddings(model, df, category_vocab, device, tokenizer):
    """Generate embeddings for all products in dataframe."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(df), 64):
            batch_df = df.iloc[start : start + 64]
            texts = batch_df["sku_name"].tolist()
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            category_ids = {
                col: torch.tensor(
                    [
                        category_vocab.get(col, {}).get(row[col], 0)
                        for _, row in batch_df.iterrows()
                    ]
                ).to(device)
                for col in CAT_COLS
            }

            emb = model(encoded["input_ids"], encoded["attention_mask"], category_ids)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 1.5 Model (Lexical Adversary)"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--m-per-class", type=int, default=4)
    parser.add_argument("--p-classes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.20e-05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--auto_timestamp", action="store_true", help="Add timestamp to output_path"
    )
    parser.add_argument("--temperature", type=float, default=0.0869)
    parser.add_argument("--triplet-weight", type=float, default=2.862)
    parser.add_argument("--class-weight", type=float, default=0.3287)
    parser.add_argument("--gate-l1-weight", type=float, default=0.0038)
    parser.add_argument("--output_path", type=str, default="runtime/checkpoints")
    parser.add_argument("--start_date", type=str, default="2025-01-01")
    parser.add_argument("--end_date", type=str, default="2025-12-31")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to cached product data (parquet)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-early-stop", action="store_true", help="Disable early stopping"
    )
    parser.add_argument(
        "--suffix-drop-rate",
        type=float,
        default=0.2492,
        help="Rate of samples to drop category suffix (0.0-1.0)",
    )
    parser.add_argument(
        "--category-dropout",
        type=float,
        default=0.2994,
        help="Probability of zeroing category embeddings during training (0.0-1.0)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1561,
        help="Ratio of warmup steps in learning rate scheduler (0.0-1.0)",
    )
    args = parser.parse_args()

    if args.auto_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_path = f"{args.output_path}_{timestamp}"
        print(f"Timestamped output path: {args.output_path}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Stage 1.5: Lexical Adversary Training")

    from training.loader import load_products, get_category_vocab
    from sklearn.model_selection import train_test_split

    df = load_products(start_date=args.start_date, end_date=args.end_date, cache_path=args.cache_path)
    print(f"Loaded {len(df)} products")

    # Always use plain sku_name (no feature engineering)
    text_column = "sku_name"

    categories = df["subclass_name"].unique()
    train_cats, val_cats = train_test_split(
        categories, test_size=0.2, random_state=args.seed
    )
    train_df = df[df["subclass_name"].isin(train_cats)].reset_index(drop=True)
    val_df = df[df["subclass_name"].isin(val_cats)].reset_index(drop=True)
    print(f"Train: {len(train_df)} products, Val: {len(val_df)} products")

    category_vocab = get_category_vocab(
        train_df
    )  # Use train_df only for strict separation
    category_vocab_sizes = {k: len(v) for k, v in category_vocab.items()}

    from models.search_model import SearchModel

    num_classes = category_vocab_sizes.get("subclass_name", 0) + 1

    model = SearchModel(
        category_vocab_sizes=category_vocab_sizes,
        embedding_dim=256,
        use_dora=True,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)
    tokenizer = model.encoder.tokenizer
    print(
        f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    print("\nPreparing HybridSampler (PK + Conflict)...")

    # No augmentation needed for plain sku_name
    train_df_aug = train_df.copy()

    dataset = PreTokenizedProductDataset(
        train_df_aug, tokenizer, category_vocab, augment=True, text_column=text_column
    )
    labels = [dataset.subclass_labels[i].item() for i in range(len(dataset))]

    sampler = HybridSampler(
        labels, train_df_aug, m=args.m_per_class, p=args.p_classes, seed=args.seed
    )

    dataloader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_products, num_workers=4
    )
    print(f"Batches per epoch: {len(sampler)}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step_scheduler = create_scheduler(
        optimizer, args.epochs * len(sampler), warmup_ratio=args.warmup_ratio
    )
    # Add ReduceLROnPlateau for when validation plateaus
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    plateau_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.output_path, exist_ok=True)

    # Save vocab
    with open(
        os.path.join(args.output_path, "category_vocab.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                col: {k: int(v) for k, v in m.items()}
                for col, m in category_vocab.items()
            },
            f,
            indent=2,
        )

    best_ndcg, best_collision, no_improve = 0.0, 0.0, 0
    patience = 999 if args.no_early_stop else 3  # Disable early stopping if flag set
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{args.epochs}\n{'=' * 50}")
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)  # Vary augmentation per epoch

        gate_mean = train_epoch(
            model,
            dataloader,
            optimizer,
            step_scheduler,
            scaler,
            device,
            triplet_weight=args.triplet_weight,
            class_weight=args.class_weight,
            gate_l1_weight=args.gate_l1_weight,
            temperature=args.temperature,
            category_dropout=args.category_dropout,
        )

        ndcg, collision = validate_with_collision(
            model, val_df, category_vocab, device, tokenizer
        )

        # Step plateau scheduler based on validation NDCG
        plateau_scheduler.step(ndcg)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Current LR: {current_lr:.2e}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ndcg": ndcg,
                    "collision": collision,
                    "epoch": epoch,
                    "gate_mean": gate_mean,
                    "category_vocab_sizes": category_vocab_sizes,
                },
                os.path.join(args.output_path, "best_model.pt"),
            )
            print("  [OK] Saved best model")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("\nEarly stopping")
                break

    print("\nLoading best model...")
    checkpoint = torch.load(
        os.path.join(args.output_path, "best_model.pt"), weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_ndcg = checkpoint.get("ndcg", 0.0)
    best_collision = checkpoint.get("collision", 0.0)

    print("Generating embeddings...")
    embeddings = generate_embeddings(model, df, category_vocab, device, tokenizer)
    np.save(os.path.join(args.output_path, "product_embeddings.npy"), embeddings)

    # === STAGE 2 COMPATIBILITY: Generate embedding_metadata.json ===
    from training.utils.data_alignment import DataAligner

    aligner = DataAligner()
    aligner.create_embedding_metadata(df, args.output_path)
    print("  [OK] Generated embedding_metadata.json for Stage 2")

    # Also save product_names_cache.json for Stage 2
    # FIX: Use dict format with sku_ids and sku_names (matching Stage 2 expectations)
    cache_data = {
        "sku_ids": df["sku_id"].tolist(),
        "sku_names": df["sku_name"].tolist(),
        "created_from_dates": f"{args.start_date} to {args.end_date}",
    }
    with open(
        os.path.join(args.output_path, "product_names_cache.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(
        f"  [OK] Saved product_names_cache.json ({len(cache_data['sku_ids'])} products)"
    )

    from inference.faiss_index import ProductIndex

    index = ProductIndex(dimension=256)
    index.build(embeddings, df["sku_id"].tolist())
    index.save(os.path.join(args.output_path, "faiss_index"))

    # Save training metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "data_range": {"start_date": args.start_date, "end_date": args.end_date},
        "metrics": {"ndcg": best_ndcg, "collision": best_collision},
        "hyperparams": vars(args),
    }
    with open(
        os.path.join(args.output_path, "training_metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n[DONE] Stage 1.5 Training complete!")
    print(f"   Total Training Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"   Best NDCG@10: {best_ndcg:.4f}")
    print(f"   Best Collision: {best_collision:.4f}")
    print(
        f"   Metadata saved to: {os.path.join(args.output_path, 'training_metadata.json')}"
    )


if __name__ == "__main__":
    main()
