from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.diagnostics.activations import ActivationMonitor
from src.diagnostics.gradients import GradientMonitor
from src.diagnostics.trainability import compute_trainability_summary
from src.models.deep_mlp import DeepMLP
from src.models.residual_mlp import ResidualBlock


def test_activation_monitor_preserves_output():
    torch.manual_seed(42)
    model = DeepMLP(in_features=32, width=64, depth=4, activation="relu", init_scheme="he")
    x = torch.randn(16, 32)

    out_plain = model(x).clone()
    mon = ActivationMonitor(model).attach()
    out_monitored = model(x).clone()
    mon.detach()

    assert torch.equal(out_plain, out_monitored)
    assert len(mon.stats) > 0


def test_gradient_monitor_preserves_gradients():
    torch.manual_seed(42)
    model = DeepMLP(in_features=32, width=64, depth=4, activation="tanh", init_scheme="xavier")
    x = torch.randn(16, 32)
    y = torch.randint(0, 10, (16,))
    crit = nn.CrossEntropyLoss()

    crit(model(x), y).backward()
    grads_plain = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()

    gmon = GradientMonitor(model).attach()
    crit(model(x), y).backward()
    grads_monitored = {n: p.grad.clone() for n, p in model.named_parameters()}
    gmon.detach()

    for n in grads_plain:
        assert torch.equal(grads_plain[n], grads_monitored[n])


def test_init_variance_matches_target():
    torch.manual_seed(123)
    # n_{l-1} * n_l = 256 * 256 = 65536 >= 10^4
    model_he = DeepMLP(in_features=256, width=256, depth=2, init_scheme="he")
    w_he = model_he.blocks[0][0].weight.detach()
    target_var_he = 2.0 / 256
    assert abs(float(w_he.var()) - target_var_he) / target_var_he <= 0.10

    model_gauss = DeepMLP(in_features=256, width=256, depth=2, init_scheme="gaussian", init_sigma=0.02)
    w_g = model_gauss.blocks[0][0].weight.detach()
    target_var_g = 0.02 ** 2
    assert abs(float(w_g.var()) - target_var_g) / target_var_g <= 0.10


def test_zero_branch_residual_block_is_identity():
    torch.manual_seed(7)
    block = ResidualBlock(width=64, alpha=1.0)
    with torch.no_grad():
        block.fc2.weight.zero_()
        block.fc2.bias.zero_()
    x = torch.randn(10, 64)
    y = block(x)
    assert torch.max(torch.abs(y - x)).item() <= 1e-6


def test_trainability_detects_nan():
    summary = compute_trainability_summary(
        final_train_loss=float("nan"),
        layer_g_history={"layer1": [1.0, float("nan")]},
        layer_variances=[1.0, 2.0],
    )
    assert summary["numerical_failure"] is True
