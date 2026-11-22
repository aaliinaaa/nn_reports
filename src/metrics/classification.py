from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Детерминированный расчёт доли верных ответов (top-1 accuracy)."""
    if targets.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return float(correct) / float(targets.numel())
