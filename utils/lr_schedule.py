"""Learning-rate schedules (no SpikingJelly / SNN imports)."""
from __future__ import annotations

import math


def cosine_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
) -> float:
    """Cosine decay after linear warmup (epoch is 0-based)."""
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(warmup_epochs, 1))
    denom = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / float(denom)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
