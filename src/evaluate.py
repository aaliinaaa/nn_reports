"""Independent evaluation entry point (section 7, E0 acceptance).

Loads a run's resolved config + checkpoint, rebuilds the model and the
*same* split (by fingerprint), and evaluates on the chosen role. This runs
separately from ``train.py``: in mode A it reproduces the final metric from
``summary.json`` to within 1e-6.

Usage:
    python -m src.evaluate --run <RUN_ID>
    python -m src.evaluate --run <RUN_ID> --role val --checkpoint best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .data import cifar
from .engine.seeds import set_global_seed
from .engine.trainer import evaluate
from .losses import build_loss
from .models.small_cnn import build_model


def _find_run(run_id: str) -> Path:
    p = Path(run_id)
    if p.exists() and (p / "manifest.json").exists():
        return p
    return Path("runs") / run_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint independently of train.")
    p.add_argument("--run", required=True, help="RUN_ID or path to a run directory")
    p.add_argument("--role", default="test", choices=["train", "val", "cal", "test"])
    p.add_argument("--checkpoint", default="last.pt", help="checkpoint filename inside runs/<id>/checkpoints/")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    run_dir = _find_run(args.run)
    cfg = Config.load(run_dir / "config.resolved.yaml")
    seed = int(cfg["experiment"]["seed"])
    deterministic = bool(cfg["experiment"]["deterministic"])
    set_global_seed(seed, deterministic=deterministic)

    split_seed = int(cfg["data"].get("split_seed", seed))
    split = cifar.make_split(
        data_root=cfg["data"]["root"],
        split_seed=split_seed,
        train_fraction=float(cfg["data"]["train_fraction"]),
    )
    bundle = cifar.get_loaders(
        data_root=cfg["data"]["root"], split=split,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=0, base_seed=seed,
    )

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model = build_model(cfg["model"]).to(device)
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    criterion = build_loss(cfg.get("losses", "name", default="cross_entropy")).to(device)

    metrics = evaluate(model, bundle["loaders"][args.role], criterion, device)
    out = {
        "run_id": run_dir.name,
        "role": args.role,
        "checkpoint": args.checkpoint,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "split_fingerprint": split.fingerprint,
    }
    (run_dir / f"evaluate_{args.role}.json").write_text(
        json.dumps(out, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
