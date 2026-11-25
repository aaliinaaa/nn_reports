from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn


class GradientMonitor:
    """Перехват градиентов по параметрам и по выходам слоёв (delta^(l)) в float64."""

    def __init__(self, model: nn.Module, layer_names: list[str] | None = None):
        self.model = model
        self.layer_names = set(layer_names) if layer_names is not None else None
        self.grad_out_stats: dict[str, dict[str, float]] = {}
        self._handles: list[Any] = []

    def attach(self) -> "GradientMonitor":
        self.detach()
        for name, module in self.model.named_modules():
            if not name:
                continue
            if self.layer_names is not None and name not in self.layer_names:
                continue
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.Linear)):
                h = module.register_full_backward_hook(self._make_bwd_hook(name))
                self._handles.append(h)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.grad_out_stats.clear()

    def _make_bwd_hook(self, name: str):
        def hook(mod: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]):
            if not grad_output or grad_output[0] is None:
                return
            with torch.no_grad():
                g64 = grad_output[0].detach().to(torch.float64)
                has_nan = bool(torch.isnan(g64).any().item() or torch.isinf(g64).any().item())
                # В PyTorch grad_output = (1/B) * dL/dh^(l) для усредненного loss по батчу;
                # умножаем на B, чтобы получить истинный delta^(l) = dL/dh^(l) из формулы (1.8)
                bs = g64.shape[0] if g64.ndim > 0 else 1
                delta = g64 * float(bs)
                delta_q = float((delta ** 2).mean().item())
                self.grad_out_stats[name] = {
                    "delta_q": delta_q,
                    "grad_out_norm": float(torch.linalg.norm(g64).item()),
                    "has_nan_or_inf": has_nan,
                }
        return hook

    @torch.no_grad()
    def collect_param_grads(self) -> dict[str, dict[str, float]]:
        """Считает норму градиента, g_l (формула 1.15) и отношение ||grad|| / ||param|| для слоёв с весами."""
        res: dict[str, dict[str, float]] = {}
        for name, module in self.model.named_modules():
            if not name or not isinstance(module, nn.Linear):
                continue
            if self.layer_names is not None and name not in self.layer_names:
                continue
            grads = [p.grad.detach().to(torch.float64).flatten() for p in module.parameters() if p.grad is not None]
            params = [p.detach().to(torch.float64).flatten() for p in module.parameters()]
            if not grads:
                continue
            g_vec = torch.cat(grads)
            p_vec = torch.cat(params)
            has_nan = bool(torch.isnan(g_vec).any().item() or torch.isinf(g_vec).any().item())
            g_norm = float(torch.linalg.norm(g_vec).item())
            p_norm = float(torch.linalg.norm(p_vec).item())
            n_params = g_vec.numel()
            g_l = g_norm / math.sqrt(max(1, n_params))
            res[name] = {
                "grad_norm": g_norm,
                "param_norm": p_norm,
                "g_l": g_l,
                "grad_to_param_ratio": g_norm / (p_norm + 1e-12),
                "num_params": int(n_params),
                "has_nan_or_inf": has_nan,
            }
        return res
