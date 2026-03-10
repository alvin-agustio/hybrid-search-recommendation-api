"""Learning rate scheduler with warmup and cosine annealing for knowledge distillation."""

import math
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """LR scheduler with linear warmup (10% steps) then cosine annealing to min_lr."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        # Linear warmup phase
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing phase
        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]
