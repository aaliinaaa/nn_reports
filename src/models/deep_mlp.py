from __future__ import annotations

import math
import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


def make_norm(name: str | None, width: int) -> nn.Module:
    if name is None or str(name).strip().lower() in ("none", "", "null"):
        return nn.Identity()
    key = str(name).strip().lower()
    if key in ("batchnorm", "bn"):
        return nn.BatchNorm1d(width)
    if key in ("layernorm", "ln"):
        return nn.LayerNorm(width)
    raise ValueError(f"Unsupported norm: {name!r}")


def init_linear_layer(layer: nn.Linear, init_scheme: str, sigma: float = 0.02) -> None:
    """Явная инициализация весов по конфигурации и нулевые смещения (раздел 8.2)."""
    fan_in = layer.in_features
    fan_out = layer.out_features
    scheme = init_scheme.strip().lower()
    if scheme == "he":
        std = math.sqrt(2.0 / fan_in)
    elif scheme == "xavier":
        std = math.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "gaussian":
        std = float(sigma)
    else:
        raise ValueError(f"Unsupported init scheme: {init_scheme!r}")

    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class DeepMLP(nn.Module):
    """Обычная полносвязная сеть глубины L (формула 1.1): Linear -> Norm -> Activation."""

    def __init__(
        self,
        *,
        in_features: int = 784,
        width: int = 256,
        depth: int = 8,
        num_classes: int = 10,
        activation: str = "relu",
        norm: str | None = "none",
        init_scheme: str = "he",
        init_sigma: float = 0.02,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.blocks = nn.ModuleList()

        cur_in = in_features
        for _ in range(depth):
            lin = nn.Linear(cur_in, width)
            init_linear_layer(lin, init_scheme, sigma=init_sigma)
            block = nn.Sequential(
                lin,
                make_norm(norm, width),
                make_activation(activation),
            )
            self.blocks.append(block)
            cur_in = width

        self.head = nn.Linear(width, num_classes)
        init_linear_layer(self.head, init_scheme, sigma=init_sigma)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = x.flatten(1)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return self.head(h)


def compute_probe_loss(model: nn.Module, x: torch.Tensor, seed: int) -> torch.Tensor:
    """Линейная пробная функция без меток L_probe (формула 1.18)."""
    h_last = model.forward_features(x)
    g = torch.Generator(device=h_last.device)
    g.manual_seed(seed)
    u = torch.randn(h_last.shape[1], generator=g, device=h_last.device, dtype=h_last.dtype)
    u = u / (torch.linalg.norm(u) + 1e-12)
    return (h_last @ u).mean()
