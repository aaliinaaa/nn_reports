from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

from src.config import resolve
from src.data.fashion_mnist import get_loaders, make_split
from src.diagnostics.activations import ActivationMonitor
from src.diagnostics.gradients import GradientMonitor
from src.diagnostics.manifest import build_manifest, save_manifest
from src.diagnostics.trainability import compute_trainability_summary
from src.engine.seeds import set_global_seed
from src.models.deep_mlp import DeepMLP, compute_probe_loss
from src.models.residual_mlp import ResidualMLP
from src.utils import git as gitutils
from src.utils.ids import compute_exp_id, compute_run_id

PROFILE_PATH = "configs/profiles/8ВМ52400.yaml"
BASE_CONFIG = "configs/lab0.yaml"
OUT_DIR = Path("reports/lab1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_harness_run(
    *,
    exp_name: str,
    model_cfg: dict,
    train_cfg: dict,
    opt_cfg: dict,
    seed: int,
    split,
    loaders: dict,
    device: torch.device,
    train_epochs: int = 0,
    probe_only: bool = False,
) -> dict:
    overrides = [
        f"experiment.name={exp_name}",
        f"experiment.seed={seed}",
        "data.name=fashion_mnist",
        f"model.name={model_cfg['arch']}",
        f"model.width={model_cfg['width']}",
        f"model.depth={model_cfg['depth']}",
        f"train.epochs={train_epochs}",
        f"train.batch_size={train_cfg['batch_size']}",
        f"optimizer.name={opt_cfg['name']}",
        f"optimizer.lr={opt_cfg['lr']}",
    ]
    for existing_sum in sorted(Path("runs").glob("*/summary.json")):
        cfg_file = existing_sum.parent / "config.resolved.yaml"
        if cfg_file.exists():
            old_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            if old_cfg.get("experiment", {}).get("name") == exp_name:
                return json.loads(existing_sum.read_text(encoding="utf-8"))

    cfg = resolve(BASE_CONFIG, profile_path=PROFILE_PATH, overrides=overrides)
    resolved = cfg._data
    resolved["lab1_model"] = model_cfg

    ginfo = gitutils.summary()
    cfg_hash = cfg.hash()
    exp_id = compute_exp_id(
        git_commit=ginfo["commit"],
        git_dirty=ginfo["dirty"],
        config_hash=cfg_hash,
        data_fingerprint=split.data_fingerprint,
        split_fingerprint=split.fingerprint,
        seed=seed,
    )
    run_id = compute_run_id(exp_id)
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "config.resolved.yaml"
    cfg.save(resolved_path)
    gitutils.save_git_diff(run_dir / "git.diff")

    set_global_seed(seed, deterministic=True)
    arch = model_cfg["arch"]
    if arch == "plain":
        model = DeepMLP(
            in_features=784,
            width=model_cfg["width"],
            depth=model_cfg["depth"],
            activation=model_cfg.get("activation", "relu"),
            norm=model_cfg.get("norm", "none"),
            init_scheme=model_cfg["init_scheme"],
            init_sigma=model_cfg.get("init_sigma", 0.02),
        ).to(device)
    else:
        alpha_mode = "one" if arch == "resnet_alpha1" else "inv_sqrt_l"
        model = ResidualMLP(
            in_features=784,
            width=model_cfg["width"],
            depth=model_cfg["depth"],
            activation=model_cfg.get("activation", "relu"),
            norm=model_cfg.get("norm", "none"),
            init_scheme=model_cfg["init_scheme"],
            init_sigma=model_cfg.get("init_sigma", 0.02),
            alpha_mode=alpha_mode,
        ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prof_block = resolved.get("profile", {})
    manifest = build_manifest(
        exp_id=exp_id,
        run_id=run_id,
        profile_id=prof_block.get("id"),
        student_id_hash=prof_block.get("student_id_hash"),
        config_hash=cfg_hash,
        data_fingerprint=split.data_fingerprint,
        split_fingerprint=split.fingerprint,
        split_sizes=split.sizes(),
        seed=seed,
        deterministic=True,
        amp=False,
        num_workers=0,
        param_count=param_count,
        resolved_config_path=str(resolved_path),
    )
    save_manifest(run_dir / "manifest.json", manifest)

    act_mon = ActivationMonitor(model).attach()
    grad_mon = GradientMonitor(model).attach()

    probe_x, probe_y = next(iter(loaders["train"]))
    probe_x, probe_y = probe_x.to(device), probe_y.to(device)

    criterion = nn.CrossEntropyLoss()
    g_history: dict[str, list[float]] = {}
    metrics_lines = []
    t_start = time.time()

    if train_epochs > 0:
        if opt_cfg["name"] == "sgd_momentum":
            optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg["lr"], momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"])

        for ep in range(train_epochs):
            model.train()
            ep_loss, ep_correct, ep_n = 0.0, 0, 0
            for step_i, (x, y) in enumerate(loaders["train"]):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                if step_i % 50 == 0:
                    pg = grad_mon.collect_param_grads()
                    for lname, st in pg.items():
                        g_history.setdefault(lname, []).append(st["g_l"])

                optimizer.step()
                ep_loss += float(loss.item()) * x.size(0)
                ep_correct += int((logits.argmax(1) == y).sum().item())
                ep_n += x.size(0)

            model.eval()
            val_loss, val_correct, val_n = 0.0, 0, 0
            with torch.no_grad():
                for vx, vy in loaders["val"]:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)
                    val_loss += float(criterion(vlogits, vy).item()) * vx.size(0)
                    val_correct += int((vlogits.argmax(1) == vy).sum().item())
                    val_n += vx.size(0)

            rec = {
                "epoch": ep,
                "train_loss": ep_loss / ep_n,
                "train_acc": ep_correct / ep_n,
                "val_loss": val_loss / val_n,
                "val_acc": val_correct / val_n,
            }
            metrics_lines.append(json.dumps(rec))

    act_mon.clear()
    grad_mon.clear()
    model.zero_grad(set_to_none=True)
    model.train()
    if probe_only:
        loss_probe = compute_probe_loss(model, probe_x, seed=seed)
        loss_probe.backward()
        final_subset_loss = float(loss_probe.item())
    else:
        logits_p = model(probe_x)
        loss_p = criterion(logits_p, probe_y)
        loss_p.backward()
        final_subset_loss = float(loss_p.item())

    param_grads = grad_mon.collect_param_grads()
    if not g_history:
        for lname, st in param_grads.items():
            g_history[lname] = [st["g_l"]]

    act_name = "Tanh" if model_cfg.get("activation", "relu") == "tanh" else "ReLU"
    act_layers = [k for k, m in model.named_modules() if m.__class__.__name__ == act_name]
    lin_layers = [k for k, m in model.named_modules() if isinstance(m, nn.Linear) and k != "head"]

    q_fwd = [act_mon.stats[k]["q"] for k in act_layers if k in act_mon.stats]
    v_fwd = [act_mon.stats[k]["var"] for k in act_layers if k in act_mon.stats]
    zero_fracs = [act_mon.stats[k]["zero_fraction"] for k in act_layers if k in act_mon.stats]
    sat_fracs = [act_mon.stats[k]["saturated_fraction"] for k in act_layers if k in act_mon.stats]
    delta_bwd = [grad_mon.grad_out_stats[k]["delta_q"] for k in act_layers if k in grad_mon.grad_out_stats]
    g_l_vals = [param_grads[k]["g_l"] for k in lin_layers if k in param_grads]

    any_nan = (
        any(s["has_nan_or_inf"] for s in act_mon.stats.values())
        or any(s["has_nan_or_inf"] for s in grad_mon.grad_out_stats.values())
        or any(s["has_nan_or_inf"] for s in param_grads.values())
    )

    trainability = compute_trainability_summary(
        final_train_loss=final_subset_loss,
        num_classes=10,
        layer_g_history=g_history,
        layer_variances=v_fwd,
        max_zero_fraction=max(zero_fracs) if zero_fracs else 0.0,
        max_saturated_fraction=max(sat_fracs) if sat_fracs else 0.0,
        has_nan_or_inf=any_nan,
    )

    act_mon.detach()
    grad_mon.detach()

    q0 = float((probe_x.flatten(1).to(torch.float64) ** 2).mean().item())
    qL = q_fwd[-1] if q_fwd else float("nan")
    c_first = None
    if arch != "plain" and hasattr(model, "blocks") and len(model.blocks) > 0:
        with torch.no_grad():
            h_in = model.stem_act(model.stem_norm(model.stem_fc(probe_x.flatten(1)))).to(torch.float64)
            blk = next(iter(model.blocks))
            br = blk.fc2(blk.act(blk.norm(blk.fc1(h_in.to(probe_x.dtype))))).to(torch.float64)
            c_first = float((br ** 2).mean().item() / ((h_in ** 2).mean().item() + 1e-30))

    last_val_acc = json.loads(metrics_lines[-1])["val_acc"] if metrics_lines else None
    last_train_acc = json.loads(metrics_lines[-1])["train_acc"] if metrics_lines else None

    summary = {
        "run_id": run_id,
        "exp_id": exp_id,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_train_time_s": time.time() - t_start,
        "best_val_acc": last_val_acc,
        "final_train_acc": last_train_acc,
        "test_acc": last_val_acc,
        "trainability": trainability,
        "layer_profiles": {
            "q_fwd": q_fwd,
            "delta_bwd": delta_bwd,
            "g_l": g_l_vals,
            "q0": q0,
            "qL": qL,
            "c_first_block": c_first,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if metrics_lines:
        (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    return summary

def estimate_chi(vals: list[float]) -> tuple[float, int]:
    """Формула (1.19): среднее геометрическое изменения за слой со 2-го скрытого слоя."""
    if len(vals) < 2:
        return float("nan"), 0
    v_start = next(v for i, v in enumerate(vals) if i == 1)
    if v_start <= 1e-30 or math.isnan(v_start) or math.isinf(v_start):
        return float("nan"), 0
    last_idx = 1
    last_val = v_start
    for i, v in enumerate(vals):
        if i <= 1:
            continue
        if v < 1e-30 or math.isnan(v) or math.isinf(v):
            break
        last_idx = i
        last_val = v
    if last_idx <= 1:
        return float("nan"), last_idx
    ratio = last_val / v_start
    return ratio ** (1.0 / (last_idx - 1)), last_idx + 1



def main():
    prof = yaml.safe_load(Path(PROFILE_PATH).read_text(encoding="utf-8"))["profile"]
    base_seed = int(prof["base_seed"])
    depths = prof["lab1_depths"]
    bad_sigma = float(prof["lab1_bad_sigma"])
    opt_name = prof["lab1_optimizer"]

    device = torch.device("cpu")
    split = make_split("./data", split_seed=base_seed, train_fraction=1.0)
    loaders = get_loaders("./data", split, batch_size=128, num_workers=0, base_seed=base_seed)["loaders"]

    # =========================================================================
    # ЭКСПЕРИМЕНТ E1: Сигнал в сети до обучения (раздел 9.1)
    # =========================================================================
    print("=== [E1] Signal propagation before training ===", flush=True)
    e1_records = []
    for d in depths:
        for scheme in ["he", "xavier", "gaussian"]:
            res = make_harness_run(
                exp_name=f"lab1_e1_d{d}_{scheme}",
                model_cfg={
                    "arch": "plain",
                    "width": 256,
                    "depth": d,
                    "activation": "relu",
                    "norm": "none",
                    "init_scheme": scheme,
                    "init_sigma": bad_sigma,
                },
                train_cfg={"batch_size": 128},
                opt_cfg={"name": opt_name, "lr": 0.01},
                seed=base_seed,
                split=split,
                loaders=loaders,
                device=device,
                train_epochs=0,
                probe_only=True,
            )
            q_fwd = res["layer_profiles"]["q_fwd"]
            d_bwd = res["layer_profiles"]["delta_bwd"]
            chi_fwd, used_f = estimate_chi(q_fwd)
            d_rev = list(reversed(d_bwd))
            chi_bwd, used_b = estimate_chi(d_rev)
            pred_chi = 1.0 if scheme == "he" else (0.5 if scheme == "xavier" else 256 * (bad_sigma ** 2) / 2.0)
            rec = {
                "run_id": res["run_id"],
                "depth": d,
                "init": scheme,
                "pred_chi": pred_chi,
                "chi_fwd": chi_fwd,
                "chi_bwd": chi_bwd,
                "used_layers_fwd": used_f,
                "used_layers_bwd": used_b,
            }
            e1_records.append(rec)
            print(
                f"  d={d:2d} {scheme:8s} -> {res['run_id']} | "
                f"pred={pred_chi:.4f} fwd={chi_fwd:.4f} bwd={chi_bwd:.4f}",
                flush=True,
            )

            if d == max(depths):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                ax1.semilogy(range(1, len(q_fwd) + 1), [max(x, 1e-38) for x in q_fwd])
                ax1.set_title(f"q^(l) (L={d}, {scheme})")
                ax1.set_xlabel("Layer l")
                ax2.semilogy(range(1, len(d_bwd) + 1), [max(x, 1e-38) for x in d_bwd])
                ax2.set_title(f"delta^(l)^2 (L={d}, {scheme})")
                ax2.set_xlabel("Layer l")
                gl = res["layer_profiles"]["g_l"]
                ax3.semilogy(range(1, len(gl) + 1), [max(x, 1e-38) for x in gl])
                ax3.set_title(f"g_l (L={d}, {scheme})")
                ax3.set_xlabel("Layer l")
                plt.tight_layout()
                plt.savefig(OUT_DIR / f"e1_profiles_d{d}_{scheme}.png", dpi=150)
                plt.close(fig)


    # Граница применимости E1: Gaussian после 1 эпохи обучения (d=24)
    e1_after_1ep = make_harness_run(
        exp_name="lab1_e1_gaussian_1epoch",
        model_cfg={
            "arch": "plain", "width": 256, "depth": 24,
            "activation": "relu", "norm": "none",
            "init_scheme": "gaussian", "init_sigma": bad_sigma,
        },
        train_cfg={"batch_size": 128},
        opt_cfg={"name": opt_name, "lr": 0.01},
        seed=base_seed, split=split, loaders=loaders, device=device,
        train_epochs=1, probe_only=True,
    )
    chi_after_1ep, _ = estimate_chi(e1_after_1ep["layer_profiles"]["q_fwd"])
    print(f"  [E1 boundary] gaussian d=24 after 1ep: {e1_after_1ep['run_id']} | chi_fwd={chi_after_1ep:.4f}", flush=True)

    # CF #2: Ширина увеличена в 4 раза (width=1024), gaussian sigma=0.02
    cf2_res = make_harness_run(
        exp_name="lab1_cf2_width1024",
        model_cfg={
            "arch": "plain", "width": 1024, "depth": 24,
            "activation": "relu", "norm": "none",
            "init_scheme": "gaussian", "init_sigma": bad_sigma,
        },
        train_cfg={"batch_size": 128},
        opt_cfg={"name": opt_name, "lr": 0.01},
        seed=base_seed, split=split, loaders=loaders, device=device,
        train_epochs=0, probe_only=True,
    )
    cf2_chi, _ = estimate_chi(cf2_res["layer_profiles"]["q_fwd"])
    print(f"  [CF2] width=1024 d=24: {cf2_res['run_id']} | pred={1024*(bad_sigma**2)/2:.4f} fwd={cf2_chi:.4f}", flush=True)

    # =========================================================================
    # ПОДБОР LR ДЛЯ E2 (раздел 9.2)
    # =========================================================================
    print("\n=== [E2 LR Tune] depth=8 plain He ===", flush=True)
    best_lr = 0.01
    best_lr_acc = -1.0
    lr_tune_records = []
    for cand_lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        r_lr = make_harness_run(
            exp_name=f"lab1_lrtune_{cand_lr}",
            model_cfg={
                "arch": "plain", "width": 256, "depth": min(depths),
                "activation": "relu", "norm": "none", "init_scheme": "he",
            },
            train_cfg={"batch_size": 128},
            opt_cfg={"name": opt_name, "lr": cand_lr},
            seed=base_seed, split=split, loaders=loaders, device=device,
            train_epochs=3, probe_only=False,
        )
        acc = r_lr["best_val_acc"] or 0.0
        lr_tune_records.append({
            "lr": cand_lr,
            "run_id": r_lr["run_id"],
            "val_acc": acc,
            "rho_T": r_lr["trainability"]["rho_T"],
        })
        print(f"  lr={cand_lr:<7g} -> {r_lr['run_id']} | val_acc={acc:.4f} rho_T={r_lr['trainability']['rho_T']:.4f}", flush=True)
        if not r_lr["trainability"]["numerical_failure"] and acc > best_lr_acc:
            best_lr_acc = acc
            best_lr = cand_lr
    print(f"Selected best_lr = {best_lr}", flush=True)

    # =========================================================================
    # ЭКСПЕРИМЕНТ E2: Глубина и остаточные связи (раздел 9.2)
    # =========================================================================
    print("\n=== [E2] Depth & Residual Connections ===", flush=True)
    e2_records = []
    e2_run_ids = []
    for arch in ["plain", "resnet_alpha1", "resnet_inv_sqrt"]:
        for d in depths:
            r = make_harness_run(
                exp_name=f"lab1_e2_{arch}_d{d}",
                model_cfg={
                    "arch": arch, "width": 256, "depth": d,
                    "activation": "relu", "norm": "none", "init_scheme": "he",
                },
                train_cfg={"batch_size": 128},
                opt_cfg={"name": opt_name, "lr": best_lr},
                seed=base_seed, split=split, loaders=loaders, device=device,
                train_epochs=3, probe_only=False,
            )
            e2_run_ids.append(r["run_id"])
            lp = r["layer_profiles"]
            tr = r["trainability"]
            q_ratio = lp["qL"] / (lp["q0"] + 1e-30)
            growth_per_block = (q_ratio ** (1.0 / d)) if (not math.isnan(q_ratio) and q_ratio > 0) else float("nan")
            rec = {
                "run_id": r["run_id"],
                "arch": arch,
                "depth": d,
                "val_acc": r["best_val_acc"],
                "rho_T": tr["rho_T"],
                "GI": tr["GI"],
                "AVD": tr["AVD"],
                "numerical_failure": tr["numerical_failure"],
                "qL_over_q0": q_ratio,
                "growth_per_block": growth_per_block,
                "c_first_block": lp["c_first_block"],
            }
            e2_records.append(rec)
            print(
                f"  {arch:15s} d={d:2d} -> {r['run_id']} | val_acc={r['best_val_acc']:.4f} "
                f"rho_T={tr['rho_T']:.4f} GI={tr['GI']:.3f} AVD={tr['AVD']:.3f} nan={tr['numerical_failure']}",
                flush=True,
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arch in ["plain", "resnet_alpha1", "resnet_inv_sqrt"]:
        sub = [x for x in e2_records if x["arch"] == arch]
        ds = [x["depth"] for x in sub]
        axes[0].plot(ds, [x["rho_T"] for x in sub], marker="o", label=arch)
        axes[1].plot(ds, [x["GI"] for x in sub], marker="o", label=arch)

    axes[0].axhline(0.5, color="green", linestyle="--", label="trainable (0.5)")
    axes[0].axhline(0.9, color="red", linestyle="--", label="untrainable (0.9)")
    axes[0].set_title("rho_T vs Depth")
    axes[0].set_xlabel("Depth L")
    axes[0].legend()

    axes[1].set_title("Gradient Imbalance (GI) vs Depth")
    axes[1].set_xlabel("Depth L")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "e2_rho_and_gi.png", dpi=150)
    plt.close(fig)


    summary_out = {
        "best_lr": best_lr,
        "lr_tune": lr_tune_records,
        "e1": e1_records,
        "e1_after_1ep": {"run_id": e1_after_1ep["run_id"], "chi_fwd": chi_after_1ep},
        "cf2": {"run_id": cf2_res["run_id"], "chi_fwd": cf2_chi},
        "e2": e2_records,
        "e2_run_ids": e2_run_ids,
    }

    (OUT_DIR / "lab1_e1_e2_results.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"\nSaved all results and plots to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()