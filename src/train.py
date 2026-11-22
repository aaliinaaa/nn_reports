"""Experiment Runner (Task 1, section 7).

Full lifecycle: config -> dataset -> model -> training -> evaluation ->
artifacts. Re-running the same experiment never overwrites a previous run
directory: a new RUN_ID is minted each time (experiment E1 relies on this).

Usage:
    python -m src.train --config configs/lab0.yaml
    python -m src.train --config configs/lab0.yaml --profile configs/profiles/<ID>.yaml
    python -m src.train --config configs/lab0.yaml --student-id 20250137
    python -m src.train --config configs/lab0.yaml --resume <RUN_ID> \\
        --override train.epochs=10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .config import resolve, parse_overrides
from .data import cifar
from .diagnostics.logger import JsonlLogger, write_summary
from .diagnostics.manifest import build_manifest, save_manifest
from .engine.checkpoint import save_checkpoint, restore_checkpoint, truncate_to_model_only
from .engine.seeds import set_global_seed
from .engine.trainer import build_optimizer, build_scheduler, train, evaluate
from .losses import build_loss
from .models.small_cnn import build_model, count_parameters
from .profile_gen import compute_profile
from .utils.git import save_git_diff, repo_root, commit_hash, is_dirty
from .utils.ids import compute_exp_id, compute_run_id


def _pick_device(cfg) -> torch.device:
    dev = cfg.get("device", "type", default="auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_run_path(run_id: str) -> Path:
    p = Path(run_id)
    if p.exists() and (p / "checkpoints" / "last.pt").exists():
        return p
    return Path("runs") / run_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a reproducible experiment.")
    p.add_argument("--config", required=True)
    p.add_argument("--profile", default=None, help="path to a per-student profile YAML")
    p.add_argument("--student-id", default=None, help="derive profile on the fly if --profile omitted")
    p.add_argument("--resume", default=None, help="RUN_ID or path to a run dir to continue from")
    p.add_argument("--truncate-checkpoint", default=None,
                   help="counterfactual #3: path to a checkpoint to strip to model-only before resume")
    p.add_argument("--override", action="append", default=[], help="key=value CLI override")
    p.add_argument("--no-test-eval", action="store_true",
                   help="skip the final test-set evaluation (test role stays unread)")
    args = p.parse_args(argv)

    repo = repo_root()

    # 1. Resolve config (base -> profile -> CLI overrides -> validate).
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
    data_root = cfg["data"]["root"]
    split_seed = int(cfg["data"].get("split_seed", seed))
    train_fraction = float(cfg["data"]["train_fraction"])
    num_workers = int(cfg["data"]["num_workers"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    amp = bool(cfg["train"]["amp"])
    save_every = int(cfg["logging"]["save_every"])

    # 2. Seed BEFORE anything that consumes randomness (model init, split, loaders).
    set_global_seed(seed, deterministic=deterministic)

    # 3. Split (reused across runs via fingerprint-keyed manifest).
    split = cifar.make_split(
        data_root=data_root, split_seed=split_seed, train_fraction=train_fraction
    )

    # 4. Data loaders with their own generators (seeded).
    bundle = cifar.get_loaders(
        data_root=data_root, split=split,
        batch_size=batch_size, num_workers=num_workers, base_seed=seed,
    )
    loaders = bundle["loaders"]
    first_batch_cksum = cifar.first_batch_checksum(loaders["train"])

    # 5. Model, optimizer, scheduler, loss, device.
    device = _pick_device(cfg)
    model = build_model(cfg["model"]).to(device)
    param_count = count_parameters(model)
    criterion = build_loss(cfg.get("losses", "name", default="cross_entropy")).to(device)
    optimizer = build_optimizer(model, cfg["optimizer"])
    steps_per_epoch = max(1, len(loaders["train"]))
    scheduler = build_scheduler(
        optimizer, cfg["scheduler"], steps_per_epoch=steps_per_epoch, epochs=epochs
    )
    scaler = torch.cuda.amp.GradScaler() if amp and device.type == "cuda" else None

    # 6. EXP_ID + RUN_ID.
    config_hash = cfg.hash()
    gcommit = commit_hash(repo)
    gdirty = is_dirty(repo)
    exp_id = compute_exp_id(
        git_commit=gcommit, git_dirty=gdirty,
        config_hash=config_hash,
        data_fingerprint=split.data_fingerprint,
        split_fingerprint=split.fingerprint,
        seed=seed,
    )
    run_id = compute_run_id(exp_id)

    # 7. Run directory; never overwrite an existing one (test #7).
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir} (mint a new RUN_ID instead)")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "plots").mkdir(parents=True)

    resolved_path = run_dir / "config.resolved.yaml"
    cfg.save(resolved_path)

    # 8. Manifest + environment.txt + git.diff (if dirty).
    profile_section = cfg.get("profile", default={}) or {}
    manifest = build_manifest(
        exp_id=exp_id, run_id=run_id,
        profile_id=profile_section.get("id"),
        student_id_hash=profile_section.get("student_id_hash"),
        config_hash=config_hash,
        data_fingerprint=split.data_fingerprint,
        split_fingerprint=split.fingerprint,
        split_sizes=split.sizes(),
        seed=seed, deterministic=deterministic, amp=amp, num_workers=num_workers,
        param_count=param_count, resolved_config_path=str(resolved_path),
        repo_root=repo,
    )
    save_manifest(run_dir / "manifest.json", manifest)
    save_git_diff(run_dir / "git.diff", repo)

    # 9. Resume (E3 / counterfactual #3).
    start_epoch, start_step = 0, 0
    resume_ckpt = None
    if args.resume:
        rd = _resolve_run_path(args.resume)
        resume_ckpt = rd / "checkpoints" / "last.pt"
        if not resume_ckpt.exists():
            raise SystemExit(f"resume checkpoint not found: {resume_ckpt}")
        if args.truncate_checkpoint:
            truncate_to_model_only(resume_ckpt, args.truncate_checkpoint)
            resume_ckpt = Path(args.truncate_checkpoint)
        meta = restore_checkpoint(
            resume_ckpt, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            map_location=device,
        )
        start_epoch = int(meta["epoch"])
        start_step = int(meta["global_step"])
        if start_epoch >= epochs:
            raise SystemExit(
                f"nothing to resume: checkpoint epoch {start_epoch} >= config epochs {epochs}"
            )

    # 10. Training loop + logging.
    with JsonlLogger(run_dir / "metrics.jsonl") as logger:
        t_total0 = time.time()
        train_result = train(
            model=model, loaders=loaders, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, device=device,
            epochs=epochs, amp=amp, save_every=save_every,
            ckpt_dir=str(run_dir / "checkpoints"), logger=logger,
            start_epoch=start_epoch, start_step=start_step, scaler=scaler,
        )
        total_train_time = time.time() - t_total0

    # 11. Final test-set evaluation (reproducible; reproduced by evaluate.py).
    test_metrics = {"loss": None, "accuracy": None}
    if not args.no_test_eval:
        test_metrics = evaluate(model, loaders["test"], criterion, device)

    # 12. Summary.
    summary = {
        "exp_id": exp_id,
        "run_id": run_id,
        "final_epoch": train_result["final_epoch"],
        "global_step": train_result["global_step"],
        "best_val_acc": train_result["best_val_acc"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
        "param_count": param_count,
        "first_batch_checksum": first_batch_cksum,
        "split_sizes": split.sizes(),
        "total_train_time_s": total_train_time,
        "resumed_from": str(resume_ckpt) if resume_ckpt else None,
        "resumed_start_epoch": start_epoch,
    }
    write_summary(run_dir / "summary.json", summary)

    print(json.dumps({"run_id": run_id, "exp_id": exp_id,
                      "test_acc": test_metrics["accuracy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
