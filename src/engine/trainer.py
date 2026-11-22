"""Training engine: optimizer/scheduler construction + the training loop.

No hyperparameter is hardcoded here (section 26): lr, weight_decay,
momentum, batch size, epochs, seed, optimizer name, scheduler name, model
size, checkpoint frequency and the split all come from the resolved config.

The loop supports:
  * AMP (off by default -- section 8);
  * warmup + cosine/step scheduler;
  * periodic checkpoints (``logging.save_every``);
  * resume from a full checkpoint without restarting from epoch 0.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..metrics import accuracy
from .checkpoint import capture_rng_state, save_checkpoint


def build_optimizer(model: nn.Module, cfg_opt: dict) -> torch.optim.Optimizer:
    name = cfg_opt.get("name", "adamw")
    lr = float(cfg_opt.get("lr", 1e-3))
    wd = float(cfg_opt.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(cfg_opt.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(optimizer, cfg_sched: dict, *, steps_per_epoch: int, epochs: int):
    name = cfg_sched.get("name", "cosine")
    warmup = int(cfg_sched.get("warmup_epochs", 0))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(0, steps_per_epoch * warmup)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "cosine":
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        if name == "step":
            gamma = float(cfg_sched.get("gamma", 0.1))
            step_size = int(cfg_sched.get("step_size", max(1, total_steps // 3)))
            return gamma ** ((step - warmup_steps) // max(1, step_size))
        raise ValueError(f"unknown scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class EpochRecord:
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    lr: float
    epoch_time: float


def _run_epoch(model, loader, criterion, device, *, train: bool, optimizer=None, scaler=None, amp: bool = False):
    model.train() if train else model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if amp and scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x); loss = criterion(logits, y)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    logits = model(x); loss = criterion(logits, y)
                    loss.backward(); optimizer.step()
            else:
                logits = model(x); loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            total_acc += accuracy(logits, y) * x.size(0)
            total_n += x.size(0)
    return total_loss / max(1, total_n), total_acc / max(1, total_n)


def train(
    *,
    model: nn.Module,
    loaders: dict[str, Any],
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    amp: bool,
    save_every: int,
    ckpt_dir: str,
    logger,
    start_epoch: int = 0,
    start_step: int = 0,
    scaler=None,
) -> dict[str, Any]:
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    steps_per_epoch = len(train_loader)
    global_step = start_step
    best_val_acc = -1.0
    best_path = f"{ckpt_dir}/best.pt"
    last_path = f"{ckpt_dir}/last.pt"

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device,
            train=True, optimizer=optimizer, scaler=scaler, amp=amp,
        )
        scheduler.step()
        global_step += steps_per_epoch
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device, train=False,
        )
        epoch_time = time.time() - t0
        lr = float(optimizer.param_groups[0]["lr"])

        rec = EpochRecord(
            epoch=epoch, global_step=global_step,
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc, lr=lr, epoch_time=epoch_time,
        )
        logger.log(rec.__dict__)
        print(f"[Epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, global_step=global_step, scaler=scaler, best_metric=best_val_acc,
            )
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(
                last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, global_step=global_step, scaler=scaler, best_metric=best_val_acc,
            )

    # Always write a final last.pt so resume/evaluate have something to load.
    save_checkpoint(
        last_path, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=epochs, global_step=global_step, scaler=scaler, best_metric=best_val_acc,
    )
    return {
        "final_epoch": epochs,
        "global_step": global_step,
        "best_val_acc": best_val_acc,
        "rng_state": capture_rng_state(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x); loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_acc += accuracy(logits, y) * x.size(0)
        total_n += x.size(0)
    return {"loss": total_loss / max(1, total_n), "accuracy": total_acc / max(1, total_n)}
