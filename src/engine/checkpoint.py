"""Full-state checkpoint manager (section 17).

A checkpoint is NOT just the model weights: it must contain every piece of
state needed to continue training from the same logical position -- optimizer,
scheduler, RNG states, epoch and global step. After resume, training must not
start over with loaded weights (test #3, experiment E3).

Counterfactual question 3 deliberately strips a checkpoint to
``model.state_dict()`` only and measures the resulting divergence.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


def capture_rng_state() -> dict[str, Any]:
    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np

    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    scaler=None,
    best_metric: float | None = None,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng": capture_rng_state(),
        "best_metric": best_metric,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, *, map_location=None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    scaler=None,
    map_location=None,
) -> dict:
    """Restore full training state. Returns the loaded metadata (epoch, step)."""
    ckpt = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt.get("rng")
    if rng:
        restore_rng_state(rng)
    return {"epoch": int(ckpt.get("epoch", 0)), "global_step": int(ckpt.get("global_step", 0)),
            "best_metric": ckpt.get("best_metric")}


def truncate_to_model_only(path: str | Path, out_path: str | Path) -> None:
    """Counterfactual #3: keep only ``model.state_dict()`` from a checkpoint."""
    ckpt = load_checkpoint(path)
    torch.save({"model": ckpt["model"]}, out_path)
