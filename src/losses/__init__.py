from __future__ import annotations

import torch.nn as nn


def build_loss(name: str = "cross_entropy") -> nn.Module:
    """Фабрика функций потерь по имени из конфигурации."""
    key = name.strip().lower()
    if key in ("cross_entropy", "ce", "crossentropy"):
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {name!r}")


__all__ = ["build_loss"]
