"""
Loss functions for semantic search training.

Includes:
- InfoNCELoss: Contrastive learning with temperature scaling (Stage 1)
- DistillationLoss: Knowledge distillation from teacher to student (Stage 2)
- HybridDistillationLoss: Combined alignment, uniformity, and ranking loss (Stage 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss."""
        embeddings = F.normalize(embeddings, p=2, dim=1)

        sim_matrix = torch.mm(embeddings, embeddings.t())  # NO temperature scaling here

        batch_size = embeddings.size(0)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pos_mask = labels_eq & ~eye_mask

        losses = []
        for i in range(batch_size):
            pos_sims = sim_matrix[i][pos_mask[i]]

            if len(pos_sims) == 0:
                continue

            # Apply temperature scaling BEFORE exp() for correct gradient flow
            pos_scaled = pos_sims / self.temperature
            all_scaled = sim_matrix[i][~eye_mask[i]] / self.temperature

            pos_logsumexp = torch.logsumexp(pos_scaled, dim=0)
            all_logsumexp = torch.logsumexp(all_scaled, dim=0)

            loss = -(pos_logsumexp - all_logsumexp)  # CORRECT InfoNCE
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()


class DistillationLoss(nn.Module):
    """Standard distillation loss (MSE on embeddings)."""

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, student_emb: torch.Tensor, teacher_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss between student and teacher embeddings."""
        # Normalize both
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)

        # MSE loss
        loss = F.mse_loss(student_norm, teacher_norm)

        return loss


class HybridDistillationLoss(nn.Module):
    """Hybrid loss combining alignment, uniformity, and ranking preservation.

    Components:
    - Alignment: cosine distance between student and teacher (primary signal)
    - Uniformity: prevents embedding collapse by spreading student embeddings
    - Ranking: preserves teacher similarity structure using hard-mined triplets
    """

    def __init__(
        self,
        alignment_weight: float = 1.0,
        ranking_weight: float = 0.4,
        uniformity_weight: float = 0.1,
        hard_negative_weight: float = 0.15,  # Kept for API compatibility, merged into ranking
        temperature: float = 0.1,
    ):
        super().__init__()
        self.alignment_weight = alignment_weight
        self.ranking_weight = ranking_weight
        self.uniformity_weight = uniformity_weight
        self.temperature = temperature

    def forward(
        self, student_emb: torch.Tensor, teacher_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute hybrid distillation loss."""
        batch_size = student_emb.size(0)

        # Normalize
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)

        # 1. Alignment loss (primary) - cosine distance between student and teacher
        alignment_loss = (
            1 - F.cosine_similarity(student_norm, teacher_norm, dim=1)
        ).mean()

        # 2. Uniformity loss (prevent collapse) - Scaled by Temperature
        sim_matrix = torch.mm(student_norm, student_norm.t()) / self.temperature
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=student_norm.device)
        uniformity_loss = torch.logsumexp(sim_matrix[mask], dim=0).mean()

        # 3. Ranking loss with teacher-guided hard triplet mining
        # Instead of sequential (i, i+1, i+2), find actual hard cases
        # using teacher similarity to identify meaningful positive/negative pairs
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        student_sim = torch.mm(student_norm, student_norm.t())

        ranking_loss = torch.tensor(0.0, device=student_norm.device)
        n_triplets = 0
        n_anchors = min(16, batch_size)

        for i in range(n_anchors):
            # Get teacher similarities for this anchor (exclude self)
            t_sims = teacher_sim[i].clone()
            t_sims[i] = -1.0  # mask self

            # Find hardest positive (most similar by teacher)
            pos_idx = t_sims.argmax().item()
            # Find hardest negative (least similar by teacher but most similar by student)
            t_sims[pos_idx] = 1.0  # mask positive
            t_sims[i] = 1.0  # mask self
            # Among remaining: find one where student wrongly ranks it high
            s_sims = student_sim[i].clone()
            s_sims[i] = -1.0
            s_sims[pos_idx] = -1.0
            # Hard negative = high student sim but low teacher sim
            neg_scores = s_sims - t_sims  # Higher = more "wrong"
            neg_idx = neg_scores.argmax().item()

            if neg_idx != i and neg_idx != pos_idx:
                student_pos = student_sim[i, pos_idx]
                student_neg = student_sim[i, neg_idx]
                # Replaced hardcoded 0.3 with Temperature for tunable margin tightness
                violation = F.relu(student_neg - student_pos + self.temperature)
                ranking_loss += violation
                n_triplets += 1

        if n_triplets > 0:
            ranking_loss = ranking_loss / n_triplets

        # Combined loss
        loss = (
            self.alignment_weight * alignment_loss
            + self.uniformity_weight * uniformity_loss
            + self.ranking_weight * ranking_loss
        )

        return loss

    def get_loss_components(
        self, student_emb: torch.Tensor, teacher_emb: torch.Tensor
    ) -> Dict[str, float]:
        """Get individual loss components for logging."""
        with torch.no_grad():
            student_norm = F.normalize(student_emb, p=2, dim=1)
            teacher_norm = F.normalize(teacher_emb, p=2, dim=1)

            alignment = (
                (1 - F.cosine_similarity(student_norm, teacher_norm, dim=1))
                .mean()
                .item()
            )

            # Uniformity: mean pairwise cosine similarity (same as forward)
            sim_matrix = torch.mm(student_norm, student_norm.t())
            mask = ~torch.eye(
                student_norm.size(0), dtype=torch.bool, device=student_norm.device
            )
            uniformity = sim_matrix[mask].mean().item()

            return {
                "alignment": alignment,
                "uniformity": uniformity,
            }
