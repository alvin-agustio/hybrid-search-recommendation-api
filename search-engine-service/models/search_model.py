"""
Stage 1 Search Model: Product Encoder + Category Embeddings + Bilinear Fusion.

Components:
- CategoryEmbedding: 5-level category hierarchy embeddings
- ProductEncoder: IndoBERT + DoRA text encoder
- BilinearFusion: Low-rank factorized fusion layer
- SearchModel: Main model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


# ============================================================================
# CATEGORY EMBEDDINGS
# ============================================================================


class CategoryEmbedding(nn.Module):
    """Learnable embeddings for 5-level category hierarchy (division/dept/class/subclass/group)."""

    def __init__(
        self,
        category_vocab_sizes: Dict[str, int],
        embedding_dim: int = 10,
        padding_idx: int = 0,
    ):
        """Initialize category embeddings with vocab sizes and dimension."""
        super().__init__()

        self.category_names = list(category_vocab_sizes.keys())
        self.embedding_dim = embedding_dim

        # Create embedding table for each category level
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=vocab_size + 1,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                )
                for name, vocab_size in category_vocab_sizes.items()
            }
        )

        # Calculate total output dimension (5 categories × 10 dim = 50)
        self.output_dim = embedding_dim * len(self.category_names)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights with small normal distribution."""
        for emb in self.embeddings.values():
            # Initialize with small random values
            nn.init.normal_(emb.weight, mean=0, std=0.02)

            # Zero out padding embeddings
            if emb.padding_idx is not None:
                emb.weight.data[emb.padding_idx].zero_()

    def forward(self, category_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed each category and concatenate into single vector."""
        embs = []

        # Embed each category level
        for name in self.category_names:
            if name in category_ids:
                emb = self.embeddings[name](category_ids[name])
                embs.append(emb)

        # Concatenate all category embeddings
        return torch.cat(embs, dim=-1)

    def get_output_dim(self) -> int:
        """Return total output dimension."""
        return self.output_dim


# ============================================================================
# BILINEAR FUSION
# ============================================================================


class BilinearFusion(nn.Module):
    """Low-rank factorized bilinear fusion (reduces params from 31M to 240K)."""

    def __init__(
        self,
        text_dim: int = 768,
        attr_dim: int = 50,
        output_dim: int = 256,
        rank: int = 256,
        dropout: float = 0.1,
        gate_bias: float = 0.0,
    ):
        """Initialize low-rank bilinear fusion with gating."""
        super().__init__()

        self.text_dim = text_dim
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        self.rank = rank
        self.gate_bias = gate_bias
        self.last_gate = None

        # Low-rank projections (text: 768→256, attr: 50→256)
        self.text_proj = nn.Linear(text_dim, rank)
        self.attr_proj = nn.Linear(attr_dim, rank)

        # Output projection if rank != output_dim
        if rank != output_dim:
            self.output_proj = nn.Linear(rank, output_dim)
        else:
            self.output_proj = None

        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual path (preserves text info)
        self.residual_proj = nn.Linear(text_dim, output_dim)

        # Gate (balances bilinear vs text)
        self.gate_proj = nn.Linear(text_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable gradient flow."""
        # Initialize projections with Xavier
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.attr_proj.weight)
        nn.init.zeros_(self.attr_proj.bias)
        nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

        # Initialize gate with bias toward text path
        nn.init.kaiming_normal_(
            self.gate_proj.weight, mode="fan_in", nonlinearity="sigmoid"
        )
        nn.init.constant_(self.gate_proj.bias, self.gate_bias)

        # Initialize output projection if exists
        if self.output_proj:
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, text_emb: torch.Tensor, attr_emb: torch.Tensor) -> torch.Tensor:
        """Fuse text and category embeddings with learnable gating."""
        # Project to low-rank space
        U = self.text_proj(text_emb)
        V = self.attr_proj(attr_emb)

        # Hadamard product (element-wise multiplication)
        fused = U * V

        # Project to output dimension if needed
        if self.output_proj:
            fused = self.output_proj(fused)

        # Apply dropout
        fused = self.dropout(fused)

        # Compute text residual
        residual = self.residual_proj(text_emb)

        # Compute gate (0-1 balance between bilinear and text)
        gate = torch.sigmoid(self.gate_proj(text_emb))

        # Store gate for L1 regularization
        self.last_gate = gate

        # Combine: output = gate * fused + (1-gate) * residual
        combined = gate * fused + (1 - gate) * residual

        # Normalize output
        output = self.layer_norm(combined)

        return output


# ============================================================================
# PRODUCT ENCODER
# ============================================================================


class ProductEncoder(nn.Module):
    """IndoBERT text encoder with DoRA for parameter-efficient fine-tuning."""

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        use_dora: bool = True,
    ):
        """Initialize IndoBERT encoder with optional DoRA."""
        super().__init__()

        self.model_name = model_name
        self.use_dora = use_dora

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)

        # Apply DoRA if enabled
        if use_dora:
            # Hardcoded optimal values (proven in production)
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.1

            # Configure DoRA
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_dora=True,
                target_modules=["query", "key", "value"],
                inference_mode=False,
            )

            # Apply DoRA to model
            self.bert = get_peft_model(self.bert, peft_config)

            # Calculate trainable params
            trainable_params = sum(
                p.numel() for p in self.bert.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.bert.parameters())

            # Print stats
            print(
                f"DoRA enabled: {trainable_params:,} / {total_params:,} trainable params ({100 * trainable_params / total_params:.2f}%)"
            )

        else:
            # Freeze all params if DoRA disabled
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "freeze_base branch triggered - DoRA disabled, freezing all params"
            )

            for param in self.bert.parameters():
                param.requires_grad = False

    def tokenize(
        self, texts: list, max_length: int = 64, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts and return input_ids and attention_mask."""
        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device if specified
        if device:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        return encoded

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract CLS token embeddings from BERT."""
        # Run BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def encode(
        self,
        texts: list,
        max_length: int = 64,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Encode texts to embeddings in batches."""
        # Get device if not specified
        if device is None:
            device = next(self.parameters()).device

        # Set to eval mode
        self.eval()
        all_embeddings = []

        # Create iterator
        iterator = range(0, len(texts), batch_size)

        # Add progress bar if requested
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Encoding")

        # Encode in batches
        with torch.no_grad():
            for i in iterator:
                # Get batch texts
                batch_texts = texts[i : i + batch_size]

                # Tokenize batch
                encoded = self.tokenize(batch_texts, max_length, device)

                # Get embeddings
                embeddings = self.forward(**encoded)

                # Move to CPU and store
                all_embeddings.append(embeddings.cpu())

        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension (768 for IndoBERT)."""
        return 768


# ============================================================================
# SEARCH MODEL
# ============================================================================


class SearchModel(nn.Module):
    """Product search model with text encoder, category embeddings, and bilinear fusion."""

    def __init__(
        self,
        category_vocab_sizes: Dict[str, int],
        use_dora: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        num_classes: int = None,
    ):
        """Initialize search model with encoder, category embeddings, and fusion."""
        super().__init__()

        self.embedding_dim = embedding_dim

        # Hardcode architectural choices (never changed in production)
        bert_model_name = "indobenchmark/indobert-base-p1"
        category_emb_dim = 10

        # Create text encoder (IndoBERT + DoRA)
        self.encoder = ProductEncoder(model_name=bert_model_name, use_dora=use_dora)
        text_dim = self.encoder.get_embedding_dim()

        # Create category embeddings
        self.category_emb = CategoryEmbedding(
            category_vocab_sizes=category_vocab_sizes, embedding_dim=category_emb_dim
        )
        attr_dim = self.category_emb.get_output_dim()

        # Create bilinear fusion layer
        self.fusion = BilinearFusion(
            text_dim=text_dim,
            attr_dim=attr_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )

        # Create projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Create classification head if requested
        self.num_classes = num_classes
        if num_classes:
            self.classification_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, num_classes),
            )
        else:
            self.classification_head = None

        # Store category vocab for encoding
        self.category_vocab = None

    def set_category_vocab(self, vocab: Dict[str, Dict[str, int]]):
        """Set category vocabulary for encoding category names to IDs."""
        self.category_vocab = vocab

    def encode_categories(
        self,
        category_values: Dict[str, List[str]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert category string values to ID tensors."""
        # Check vocab is set
        if self.category_vocab is None:
            raise ValueError("Category vocab not set. Call set_category_vocab() first.")

        category_ids = {}

        # Convert each category value to ID
        for name, values in category_values.items():
            if name in self.category_vocab:
                # Map values to IDs (0 for unknown)
                ids = [self.category_vocab[name].get(v, 0) for v in values]

                # Create tensor
                tensor = torch.tensor(ids, dtype=torch.long)

                # Move to device if specified
                if device:
                    tensor = tensor.to(device)

                category_ids[name] = tensor

        return category_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_ids: Dict[str, torch.Tensor],
        return_logits: bool = False,
    ):
        """Encode inputs to L2-normalized embeddings."""
        # Encode text
        text_emb = self.encoder(input_ids, attention_mask)

        # Encode categories
        attr_emb = self.category_emb(category_ids)

        # Fuse text and categories
        fused = self.fusion(text_emb, attr_emb)

        # Project to final dimension
        output = self.projection(fused)

        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=-1)

        # Return with classification logits if requested
        if return_logits and self.classification_head is not None:
            logits = self.classification_head(output)
            return output, logits

        return output

    def encode_products(
        self,
        product_names: List[str],
        category_values: Dict[str, List[str]],
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode products to embeddings in batches."""
        # Get device if not specified
        if device is None:
            device = next(self.parameters()).device

        # Set to eval mode
        self.eval()
        all_embeddings = []

        # Calculate number of products
        n_products = len(product_names)
        iterator = range(0, n_products, batch_size)

        # Add progress bar if requested
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Encoding products")

        # Encode in batches
        with torch.no_grad():
            for i in iterator:
                # Get batch end index
                batch_end = min(i + batch_size, n_products)

                # Get batch data
                batch_names = product_names[i:batch_end]
                batch_categories = {
                    name: values[i:batch_end]
                    for name, values in category_values.items()
                }

                # Tokenize text
                text_encoded = self.encoder.tokenize(batch_names, device=device)

                # Encode categories
                cat_ids = self.encode_categories(batch_categories, device=device)

                # Forward pass
                embeddings = self.forward(
                    input_ids=text_encoded["input_ids"],
                    attention_mask=text_encoded["attention_mask"],
                    category_ids=cat_ids,
                )

                # Store on CPU
                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        return np.vstack(all_embeddings)

    def encode_query(
        self,
        query: str,
        default_category_ids: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Encode search query with default category embeddings."""
        # Get device if not specified
        if device is None:
            device = next(self.parameters()).device

        # Set to eval mode
        self.eval()

        with torch.no_grad():
            # Tokenize query
            text_encoded = self.encoder.tokenize([query], device=device)

            # Use zero category embeddings for queries (no category info)
            if default_category_ids is None:
                default_category_ids = {
                    name: torch.zeros(1, dtype=torch.long, device=device)
                    for name in self.category_emb.category_names
                }

            # Forward pass
            embedding = self.forward(
                input_ids=text_encoded["input_ids"],
                attention_mask=text_encoded["attention_mask"],
                category_ids=default_category_ids,
            )

        return embedding.cpu().numpy()
