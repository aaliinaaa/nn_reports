from __future__ import annotations

import math
import statistics
from typing import Any


def compute_trainability_summary(
    *,
    final_train_loss: float,
    num_classes: int = 10,
    layer_g_history: dict[str, list[float]] | None = None,
    layer_variances: list[float] | None = None,
    max_zero_fraction: float = 0.0,
    max_saturated_fraction: float = 0.0,
    has_nan_or_inf: bool = False,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Вычисляет показатели обучаемости (формулы 1.14 - 1.17) для summary.json."""
    numerical_failure = bool(
        has_nan_or_inf
        or math.isnan(final_train_loss)
        or math.isinf(final_train_loss)
    )

    rho_T = float(final_train_loss / math.log(max(2, num_classes))) if not numerical_failure else float("nan")

    # GI (формула 1.16): по медианам g_l по шагам обучения
    gi = float("nan")
    if layer_g_history:
        medians = []
        for vals in layer_g_history.values():
            clean = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
            if clean:
                medians.append(statistics.median(clean))
            else:
                numerical_failure = True
        if medians and not numerical_failure:
            gi = math.log10((max(medians) + eps) / (min(medians) + eps))

    # AVD (формула 1.17): изменение разброса активаций с глубиной
    avd = float("nan")
    if layer_variances:
        if any(math.isnan(v) or math.isinf(v) for v in layer_variances):
            numerical_failure = True
        else:
            v1 = layer_variances[0]
            avd = max(abs(math.log10(vl + eps) - math.log10(v1 + eps)) for vl in layer_variances)

    return {
        "rho_T": rho_T,
        "GI": gi,
        "AVD": avd,
        "max_zero_fraction": float(max_zero_fraction),
        "max_saturated_fraction": float(max_saturated_fraction),
        "numerical_failure": numerical_failure,
    }
