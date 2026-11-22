"""Model profiler CLI (Task 6, section 19, experiment E4).

Measures trainable parameter count, forward-pass time, per-batch latency,
throughput (samples/s), one-epoch wall time, and peak GPU memory when CUDA
is available. Latency is measured after warm-up; on CUDA the device is
synchronized around each timed interval (section 22.5).

Usage:
    python -m src.profile --config configs/lab0.yaml
    python -m src.profile --config configs/lab0.yaml --student-id 20250137
    python -m src.profile --config configs/lab0.yaml --batch-sizes 1 8 32 128
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import resolve
from .data import cifar
from .diagnostics.profiler import profile_model, estimate_epoch_time
from .engine.seeds import set_global_seed
from .losses import build_loss
from .models.small_cnn import build_model
from .profile_gen import compute_profile


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile the model: params, latency, throughput, memory.")
    p.add_argument("--config", required=True)
    p.add_argument("--profile", default=None)
    p.add_argument("--student-id", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--epoch-max-batches", type=int, default=None,
                   help="if set, estimate epoch time from a prefix and extrapolate")
    args = p.parse_args(argv)

    profile_path = args.profile
    if profile_path is None and args.student_id:
        prof = compute_profile(args.student_id)
        os.makedirs("configs/profiles", exist_ok=True)
        profile_path = f"configs/profiles/{args.student_id}.yaml"
        import yaml
        Path(profile_path).write_text(
            yaml.safe_dump(prof, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    cfg = resolve(args.config, profile_path=profile_path, overrides=args.override)

    seed = int(cfg["experiment"]["seed"])
    deterministic = bool(cfg["experiment"]["deterministic"])
    set_global_seed(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_sizes = args.batch_sizes or cfg.get("profile", "profile_batch_sizes", default=[1, 8, 32, 128])

    # Build a model and a real train loader for the epoch-time measurement.
    model = build_model(cfg["model"])
    split_seed = int(cfg["data"].get("split_seed", seed))
    split = cifar.make_split(
        data_root=cfg["data"]["root"], split_seed=split_seed,
        train_fraction=float(cfg["data"]["train_fraction"]),
    )
    bundle = cifar.get_loaders(
        data_root=cfg["data"]["root"], split=split,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]), base_seed=seed,
    )
    criterion = build_loss(cfg.get("losses", "name", default="cross_entropy")).to(device)

    input_shape = (3, 32, 32)
    profile = profile_model(
        model=model, input_shape=input_shape, device=device,
        batch_sizes=batch_sizes, warmup=args.warmup, repeats=args.repeats,
    )

    epoch_time = estimate_epoch_time(
        model=model.to(device), loader=bundle["loaders"]["train"], device=device,
        criterion=criterion, amp=bool(cfg["train"]["amp"]),
        max_batches=args.epoch_max_batches,
    )
    profile["epoch_time_s"] = epoch_time
    profile["epochs_in_config"] = int(cfg["train"]["epochs"])
    profile["estimated_full_run_s"] = epoch_time * int(cfg["train"]["epochs"])

    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports") / f"profile_{profile['param_count']}_{device.type}.json"
    out_path.write_text(json.dumps(profile, sort_keys=True, indent=2), encoding="utf-8")

    print(json.dumps({
        "param_count": profile["param_count"],
        "device": profile["device"],
        "epoch_time_s": round(epoch_time, 4),
        "estimated_full_run_s": round(profile["estimated_full_run_s"], 2),
        "per_batch_size": [
            {"bs": r["batch_size"],
             "latency_ms_mean": round(r["latency_s"]["mean"] * 1e3, 4),
             "latency_ms_p95": round(r["latency_s"]["p95"] * 1e3, 4),
             "throughput_sps_mean": round(r["throughput_sps"]["mean"], 2),
             "peak_gpu_memory_bytes": r["peak_gpu_memory_bytes"]}
            for r in profile["per_batch_size"]
        ],
        "report": str(out_path),
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
