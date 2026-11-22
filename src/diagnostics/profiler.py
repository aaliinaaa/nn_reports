"""Model profiler (section 19, experiment E4).

Measures: trainable parameter count, average forward-pass time, per-batch
latency, throughput (samples/s), training-epoch wall time, and peak GPU
memory when CUDA is available. Latency is measured after warm-up; for CUDA
the device is synchronized before and after each timed interval.

Reported statistics: mean, median, p95 (section 22.5).
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Any

import torch
import torch.nn as nn

from ..models.small_cnn import count_parameters
from ..metrics import accuracy


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"),
                "cv": float("nan"), "n": 0}
    s = sorted(samples)
    n = len(s)
    mean = statistics.fmean(s)
    var = statistics.fmean((x - mean) ** 2 for x in s)
    std = math.sqrt(var)
    p95 = s[min(n - 1, int(0.95 * n))]
    cv = std / mean if mean > 0 else float("nan")
    return {"mean": mean, "median": s[n // 2], "p95": p95, "cv": cv, "n": n}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_model(
    *,
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    batch_sizes: list[int],
    warmup: int = 10,
    repeats: int = 30,
    criterion=None,
) -> dict[str, Any]:
    """Return latency/throughput statistics per batch size plus global info."""
    model = model.to(device).eval()
    param_count = count_parameters(model)
    results: list[dict[str, Any]] = []

    for bs in batch_sizes:
        x = torch.randn(bs, *input_shape, device=device)
        y = torch.randint(0, 10, (bs,), device=device)

        # Warm-up (not timed): first iterations allocate kernels/buffers.
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
        _sync(device)

        latencies: list[float] = []
        peak_mem_before = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for _ in range(repeats):
                _sync(device)
                t0 = time.perf_counter()
                logits = model(x)
                _sync(device)
                t1 = time.perf_counter()
                latencies.append(t1 - t0)

        peak_mem = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        lat_stats = _stats(latencies)
        throughput = [
            bs / lat for lat in latencies if lat > 0
        ]
        tp_stats = _stats(throughput)
        results.append({
            "batch_size": bs,
            "latency_s": lat_stats,
            "throughput_sps": tp_stats,
            "peak_gpu_memory_bytes": peak_mem,
            "forward_accuracy_check": float(accuracy(logits, y)) if criterion is None else None,
        })

    # Forward pass time for a single sample (batch=1), reported separately.
    single = next((r for r in results if r["batch_size"] == 1), results[0] if results else None)

    return {
        "param_count": param_count,
        "device": str(device),
        "input_shape": list(input_shape),
        "warmup_iters": warmup,
        "repeats": repeats,
        "per_batch_size": results,
        "forward_pass_single_batch_s": single["latency_s"] if single else None,
    }


def estimate_epoch_time(*, model, loader, device, criterion, amp: bool, max_batches: int | None = None) -> float:
    """Measure wall time of one training epoch (or its prefix for a quick estimate)."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # dummy, no real update needed
    scaler = torch.cuda.amp.GradScaler() if amp and device.type == "cuda" else None
    _sync(device)
    t0 = time.perf_counter()
    n = 0
    for x, y in loader:
        if max_batches is not None and n >= max_batches:
            break
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        n += 1
    _sync(device)
    elapsed = time.perf_counter() - t0
    total_batches = len(loader)
    if max_batches is not None and n < total_batches:
        return elapsed * (total_batches / max(1, n))
    return elapsed
