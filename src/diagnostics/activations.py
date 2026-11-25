from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


class ActivationMonitor:
    """Перехват активаций слоёв через forward hooks (все статистики в float64)."""

    def __init__(self, model: nn.Module, layer_names: list[str] | None = None):
        self.model = model
        self.layer_names = set(layer_names) if layer_names is not None else None
        self.stats: dict[str, dict[str, float]] = {}
        self._handles: list[Any] = []
        self._pre_act_cache: dict[str, torch.Tensor] = {}

    def attach(self) -> "ActivationMonitor":
        self.detach()
        for name, module in self.model.named_modules():
            if not name:
                continue
            if self.layer_names is not None and name not in self.layer_names:
                continue
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.Linear)):
                h = module.register_forward_hook(self._make_hook(name, module))
                self._handles.append(h)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.stats.clear()
        self._pre_act_cache.clear()

    def _make_hook(self, name: str, module: nn.Module):
        def hook(mod: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            with torch.no_grad():
                out64 = output.detach().to(torch.float64)
                has_nan = bool(torch.isnan(out64).any().item() or torch.isinf(out64).any().item())
                mean = float(out64.mean().item())
                std = float(out64.std(unbiased=False).item())
                var = float(out64.var(unbiased=False).item())
                q = float((out64 ** 2).mean().item())
                vmin = float(out64.min().item())
                vmax = float(out64.max().item())

                zero_frac = float((out64 == 0.0).to(torch.float64).mean().item()) if isinstance(mod, nn.ReLU) else 0.0
                sat_frac = 0.0
                if isinstance(mod, nn.Tanh) and len(inputs) > 0:
                    z64 = inputs[0].detach().to(torch.float64)
                    # |tanh'(z)| = |1 - tanh^2(z)| < 0.1
                    deriv = 1.0 - torch.tanh(z64) ** 2
                    sat_frac = float((deriv.abs() < 0.1).to(torch.float64).mean().item())

                self.stats[name] = {
                    "mean": mean,
                    "std": std,
                    "var": var,
                    "q": q,
                    "min": vmin,
                    "max": vmax,
                    "zero_fraction": zero_frac,
                    "saturated_fraction": sat_frac,
                    "has_nan_or_inf": has_nan,
                }
        return hook
