"""Enhanced query encoder with attention pooling and residual projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer
from typing import Optional, List
import math


class AttentionPooling(nn.Module):
    """Multi-head attention pooling for better sequence representation."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        """Initialize attention pooling with multi-head attention."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q, K, V projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Regularization
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool sequence using multi-head attention."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Project to Q, K, V and reshape for multi-head
        Q = self.query(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        K = self.key(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        V = self.value(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        )

        # Global average pooling with attention weights
        attn_mean = attention_weights.mean(dim=1).mean(dim=-1)
        attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)
        pooled = torch.sum(context * attn_mean.unsqueeze(-1), dim=1)

        # Normalize output
        pooled = self.layer_norm(pooled)

        return pooled


class ResidualProjection(nn.Module):
    """Residual projection head with skip connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize residual projection with multiple layers."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [self._make_residual_block(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _make_residual_block(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Create residual block with layer norm and dropout."""
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply residual blocks
        for block in self.blocks:
            residual = x
            x = block(x) + residual

        # Project to output dimension
        x = self.output_proj(x)
        x = self.layer_norm(x)

        return x


class EnhancedQueryEncoder(nn.Module):
    """Enhanced query encoder with attention pooling and residual projection."""

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-lite-base-p1",
        output_dim: int = 256,
        freeze_base: bool = False,
        use_attention_pooling: bool = True,
        use_adapters: bool = False,
    ):
        """Initialize enhanced query encoder."""
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_attention_pooling = use_attention_pooling
        self.use_adapters = use_adapters

        # Hardcoded optimal values (proven in production)
        hidden_dim = 512
        num_projection_layers = 3
        num_attention_heads = 8
        dropout = 0.1

        # Load base encoder (try offline first)
        try:
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except (OSError, EnvironmentError):
            # Fallback to network if local files not found
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Get hidden size from encoder config
        hidden_size = self.encoder.config.hidden_size

        # Create pooling layer
        if use_attention_pooling:
            self.pooling = AttentionPooling(hidden_size, num_attention_heads)
        else:
            self.pooling = None

        # Create projection head
        self.projection = ResidualProjection(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_projection_layers,
            dropout=dropout,
        )

        # Create adapter layers if requested
        if use_adapters:
            self.adapters = self._create_adapters(hidden_size)
        else:
            self.adapters = None

        # Freeze base encoder if requested
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Print parameter count
        self._print_param_count()

    def _create_adapters(self, hidden_size: int) -> nn.ModuleList:
        """Create adapter layers for efficient fine-tuning."""
        adapter_size = 64
        adapters = nn.ModuleList()

        # Add adapter to each transformer layer
        for _ in range(self.encoder.config.num_hidden_layers):
            adapter = nn.Sequential(
                nn.Linear(hidden_size, adapter_size),
                nn.GELU(),
                nn.Linear(adapter_size, hidden_size),
                nn.Dropout(0.1),
            )
            adapters.append(adapter)

        return adapters

    def _print_param_count(self):
        """Print parameter count for debugging."""
        # Calculate total and trainable params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EnhancedQueryEncoder: {trainable:,} / {total:,} trainable params")

        # Break down by component
        encoder_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        projection_params = sum(p.numel() for p in self.projection.parameters())

        print(f"  - Encoder: {encoder_params:,} trainable")
        print(f"  - Projection: {projection_params:,}")

        # Print pooling params if exists
        if self.pooling:
            pooling_params = sum(p.numel() for p in self.pooling.parameters())
            print(f"  - Attention Pooling: {pooling_params:,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input to L2-normalized embeddings."""
        # Get encoder outputs
        # Only output all hidden states if adapters are enabled (saves memory/computation)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.adapters is not None,
        )

        # Apply adapters if enabled
        if self.adapters is not None:
            hidden_states = outputs.hidden_states[-1]
            for i, adapter in enumerate(self.adapters):
                if i < len(outputs.hidden_states) - 1:
                    layer_output = outputs.hidden_states[i + 1]
                    adapted = adapter(layer_output)
                    hidden_states = hidden_states + adapted
        else:
            hidden_states = outputs.last_hidden_state

        # Pool sequence
        if self.use_attention_pooling:
            pooled = self.pooling(hidden_states, attention_mask)
        else:
            # Use CLS token
            pooled = hidden_states[:, 0, :]

        # Project to output dimension
        projected = self.projection(pooled)

        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)

        return normalized

    def encode(
        self,
        text: str,
        device: Optional[torch.device] = None,
        max_length: int = 64,
    ) -> torch.Tensor:
        """Encode single query text to embedding."""
        # Get device if not specified
        if device is None:
            device = next(self.parameters()).device

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            embedding = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        return embedding.squeeze(0)

    def encode_batch(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        max_length: int = 64,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode multiple texts to embeddings in batches."""
        # Get device if not specified
        if device is None:
            device = next(self.parameters()).device

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            # Get batch texts
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Encode batch
            with torch.no_grad():
                embeddings = self.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            # Store on CPU
            all_embeddings.append(embeddings.cpu())

        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def save(self, path: str, scaler=None):
        """Save model checkpoint with AMP scaler state if provided."""
        import os

        # Create directory
        os.makedirs(path, exist_ok=True)

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "output_dim": self.output_dim,
            "use_attention_pooling": self.use_attention_pooling,
            "use_adapters": self.use_adapters,
        }

        # Save scaler state if AMP enabled
        if scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()
            checkpoint_data["amp_enabled"] = True
        else:
            checkpoint_data["amp_enabled"] = False

        # Save model state and config
        torch.save(
            checkpoint_data,
            os.path.join(path, "enhanced_query_encoder.pt"),
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

        print(f"Enhanced model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Load model from checkpoint."""
        import os

        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(path, "enhanced_query_encoder.pt"),
            map_location=device or "cpu",
        )

        # Create model with saved config
        model = cls(
            model_name=checkpoint["model_name"],
            output_dim=checkpoint["output_dim"],
            use_attention_pooling=checkpoint.get("use_attention_pooling", True),
            use_adapters=checkpoint.get("use_adapters", False),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move to device if specified
        if device:
            model = model.to(device)

        # Return scaler state if present (for AMP resume)
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        amp_enabled = checkpoint.get("amp_enabled", False)

        print(f"Enhanced model loaded from {path}")
        if scaler_state_dict:
            print(f"[*] AMP scaler state found in checkpoint")

        return model, scaler_state_dict, amp_enabled
