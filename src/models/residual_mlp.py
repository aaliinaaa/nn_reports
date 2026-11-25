from __future__ import annotations

import math
import torch
import torch.nn as nn

from .deep_mlp import init_linear_layer, make_activation, make_norm


class ResidualBlock(nn.Module):
    """Остаточный блок (формула 1.11): h^(l+1) = h^(l) + alpha * F_l(h^(l))."""

    def __init__(
        self,
        width: int,
        activation: str = "relu",
        norm: str | None = "none",
        init_scheme: str = "he",
        init_sigma: float = 0.02,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.fc1 = nn.Linear(width, width)
        self.norm = make_norm(norm, width)
        self.act = make_activation(activation)
        self.fc2 = nn.Linear(width, width)

        init_linear_layer(self.fc1, init_scheme, sigma=init_sigma)
        init_linear_layer(self.fc2, init_scheme, sigma=init_sigma)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        branch = self.fc2(self.act(self.norm(self.fc1(h))))
        return h + self.alpha * branch


class ResidualMLP(nn.Module):
    """Остаточная сеть из L блоков (всего 2L линейных слоёв в остаточных блоках)."""

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
        alpha_mode: str = "one",  # "one" (alpha=1) | "inv_sqrt_l" (alpha=1/sqrt(L)) | float
    ):
        super().__init__()
        self.depth = depth
        self.width = width

        if isinstance(alpha_mode, (int, float)):
            alpha = float(alpha_mode)
        elif str(alpha_mode).strip().lower() in ("one", "1"):
            alpha = 1.0
        elif str(alpha_mode).strip().lower() in ("inv_sqrt_l", "scaled"):
            alpha = 1.0 / math.sqrt(max(1, depth))
        else:
            raise ValueError(f"Unsupported alpha_mode: {alpha_mode!r}")

        self.alpha = alpha
        self.stem_fc = nn.Linear(in_features, width)
        self.stem_norm = make_norm(norm, width)
        self.stem_act = make_activation(activation)
        init_linear_layer(self.stem_fc, init_scheme, sigma=init_sigma)

        self.blocks = nn.ModuleList([
            ResidualBlock(
                width=width,
                activation=activation,
                norm=norm,
                init_scheme=init_scheme,
                init_sigma=init_sigma,
                alpha=alpha,
            )
            for _ in range(depth)
        ])

        self.head = nn.Linear(width, num_classes)
        init_linear_layer(self.head, init_scheme, sigma=init_sigma)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem_act(self.stem_norm(self.stem_fc(x.flatten(1))))
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
