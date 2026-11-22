"""Compact CNN baseline (section 13).

The model is intentionally small and fast: the lab studies reproducibility,
not SOTA accuracy. Width and depth come from the config (the profile only
changes *configuration*, never *code* -- test #10).
"""
from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    """Stack of ``depth`` conv blocks with channel growth + global pooling."""

    def __init__(self, *, width: int = 64, depth: int = 4, num_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        # Channel width doubles every two blocks until a cap; depth controls
        # the number of conv blocks.
        for i in range(depth):
            out_ch = min(width * (2 ** (i // 2)), width * 4)
            layers.append(ConvBlock(in_ch, out_ch))
            # Downsample every other block with stride-2 max-pool.
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(cfg_model: dict) -> nn.Module:
    return SmallCNN(
        width=int(cfg_model.get("width", 64)),
        depth=int(cfg_model.get("depth", 4)),
        num_classes=int(cfg_model.get("num_classes", 10)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
