from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from run_lab1_e1_e2 import BASE_CONFIG, OUT_DIR, PROFILE_PATH, make_harness_run
from src.data.fashion_mnist import get_loaders, make_split


def check_hypothesis_committed() -> None:
    hyp_path = Path("hypothesis.md")
    if not hyp_path.exists():
        raise FileNotFoundError(
            "Файл hypothesis.md не найден! Создайте его и закоммитьте в Git до запуска E3."
        )
    res = subprocess.run(
        ["git", "status", "--porcelain", "hypothesis.md"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0 and res.stdout.strip() != "":
        raise RuntimeError(
            "Файл hypothesis.md имеет незакоммиченные изменения! "
            "Выполните `git add hypothesis.md && git commit -m 'Add E3 hypothesis'` перед запуском E3."
        )


def build_trainability_map(e2_records: list[dict], out_path: Path) -> None:
    """Строит двумерный срез карты обучаемости: глубина x архитектура -> rho_T."""
    archs = ["plain", "resnet_alpha1", "resnet_inv_sqrt"]
    depths = sorted({r["depth"] for r in e2_records})
    grid = np.full((len(archs), len(depths)), np.nan)

    for r in e2_records:
        i = archs.index(r["arch"])
        j = depths.index(r["depth"])
        grid[i, j] = r["rho_T"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#d3d3d3")

    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs)
    ax.set_xlabel("Depth L")
    ax.set_ylabel("Architecture")
    ax.set_title("Trainability Map: rho_T (width=256, He, ReLU, no norm, lr=0.01)")

    for i in range(len(archs)):
        for j in range(len(depths)):
            val = grid[i, j]
            txt = "NaN\n(fail)" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("rho_T (<=0.5 trainable, >=0.9 untrainable)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    check_hypothesis_committed()

    prof = yaml.safe_load(Path(PROFILE_PATH).read_text(encoding="utf-8"))["profile"]
    base_seed = int(prof["base_seed"])
    depths = prof["lab1_depths"]
    opt_name = prof["lab1_optimizer"]

    e1_e2_file = OUT_DIR / "lab1_e1_e2_results.json"
    e1_e2_data = json.loads(e1_e2_file.read_text(encoding="utf-8"))
    best_lr = float(e1_e2_data["best_lr"])

    device = torch.device("cpu")
    split = make_split("./data", split_seed=base_seed, train_fraction=1.0)
    loaders = get_loaders("./data", split, batch_size=128, num_workers=0, base_seed=base_seed)["loaders"]

    # =========================================================================
    # 1. Дозамер E2 (п. 1 разд. 9.2): q^(L)/q^(0) и c ДО обучения (train_epochs=0)
    # =========================================================================
    print("=== [E2 Pre-train Check] ResNet signal growth before training ===", flush=True)
    pretrain_resnet_records = []
    for arch in ["resnet_alpha1", "resnet_inv_sqrt"]:
        for d in depths:
            r = make_harness_run(
                exp_name=f"lab1_e2_pretrain_{arch}_d{d}",
                model_cfg={
                    "arch": arch,
                    "width": 256,
                    "depth": d,
                    "activation": "relu",
                    "norm": "none",
                    "init_scheme": "he",
                },
                train_cfg={"batch_size": 128},
                opt_cfg={"name": opt_name, "lr": best_lr},
                seed=base_seed,
                split=split,
                loaders=loaders,
                device=device,
                train_epochs=0,
                probe_only=True,
            )
            lp = r["layer_profiles"]
            q_ratio = lp["qL"] / (lp["q0"] + 1e-30)
            c_val = lp["c_first_block"]
            growth_per_block = (q_ratio ** (1.0 / d)) if (not math.isnan(q_ratio) and q_ratio > 0) else float("nan")
            theory_1_plus_c = (1.0 + c_val) if c_val is not None else float("nan")
            theory_bound_inv_sqrt = (1.2 * math.exp(c_val)) if c_val is not None else float("nan")

            rec = {
                "run_id": r["run_id"],
                "arch": arch,
                "depth": d,
                "qL_over_q0": q_ratio,
                "c_first_block": c_val,
                "measured_growth_per_block": growth_per_block,
                "theory_1_plus_c": theory_1_plus_c,
                "bound_1_2_exp_c": theory_bound_inv_sqrt,
            }
            pretrain_resnet_records.append(rec)
            print(
                f"  {arch:15s} d={d:2d} -> {r['run_id']} | qL/q0={q_ratio:.4e} "
                f"c={c_val:.4f} (qL/q0)^(1/L)={growth_per_block:.4f} (1+c={theory_1_plus_c:.4f})",
                flush=True,
            )

    # =========================================================================
    # 2. Карта обучаемости (раздел 12, п. 4)
    # =========================================================================
    map_path = OUT_DIR / "trainability_map.png"
    build_trainability_map(e1_e2_data["e2"], map_path)
    print(f"\nSaved trainability map to {map_path}", flush=True)

    # =========================================================================
    # 3. Эксперимент E3: Основное сравнение на 3 seeds (раздел 9.3)
    # =========================================================================
    print("\n=== [E3] Main comparison across seeds s = 1, 2, 3 (depth=48) ===", flush=True)
    e3_depth = 48
    e3_seeds = [1, 2, 3]
    e3_runs = {"A_resnet_inv_sqrt": [], "B_plain": []}

    for s in e3_seeds:
        r_a = make_harness_run(
            exp_name=f"lab1_e3_A_resnet_inv_sqrt_d{e3_depth}_s{s}",
            model_cfg={
                "arch": "resnet_inv_sqrt",
                "width": 256,
                "depth": e3_depth,
                "activation": "relu",
                "norm": "none",
                "init_scheme": "he",
            },
            train_cfg={"batch_size": 128},
            opt_cfg={"name": opt_name, "lr": best_lr},
            seed=s,
            split=split,
            loaders=loaders,
            device=device,
            train_epochs=3,
            probe_only=False,
        )
        r_b = make_harness_run(
            exp_name=f"lab1_e3_B_plain_d{e3_depth}_s{s}",
            model_cfg={
                "arch": "plain",
                "width": 256,
                "depth": e3_depth,
                "activation": "relu",
                "norm": "none",
                "init_scheme": "he",
            },
            train_cfg={"batch_size": 128},
            opt_cfg={"name": opt_name, "lr": best_lr},
            seed=s,
            split=split,
            loaders=loaders,
            device=device,
            train_epochs=3,
            probe_only=False,
        )

        e3_runs["A_resnet_inv_sqrt"].append({
            "seed": s,
            "run_id": r_a["run_id"],
            "val_acc": r_a["best_val_acc"],
            "rho_T": r_a["trainability"]["rho_T"],
            "GI": r_a["trainability"]["GI"],
            "AVD": r_a["trainability"]["AVD"],
        })
        e3_runs["B_plain"].append({
            "seed": s,
            "run_id": r_b["run_id"],
            "val_acc": r_b["best_val_acc"],
            "rho_T": r_b["trainability"]["rho_T"],
            "GI": r_b["trainability"]["GI"],
            "AVD": r_b["trainability"]["AVD"],
        })
        print(
            f"  seed={s} | A (resnet_inv_sqrt): val_acc={r_a['best_val_acc']:.4f}, rho_T={r_a['trainability']['rho_T']:.4f} ({r_a['run_id']})\n"
            f"         | B (plain):           val_acc={r_b['best_val_acc']:.4f}, rho_T={r_b['trainability']['rho_T']:.4f} ({r_b['run_id']})",
            flush=True,
        )

    # Расчёт статистик по формуле (1.20) для val_acc и rho_T
    stats_out = {}
    for metric in ["val_acc", "rho_T"]:
        deltas = [
            a[metric] - b[metric]
            for a, b in zip(e3_runs["A_resnet_inv_sqrt"], e3_runs["B_plain"])
        ]
        mean_delta = sum(deltas) / len(deltas)
        std_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / (len(deltas) - 1))
        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        established = same_sign and (abs(mean_delta) >= 2.0 * std_delta)

        stats_out[metric] = {
            "deltas": deltas,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "two_std": 2.0 * std_delta,
            "same_sign": same_sign,
            "difference_established": established,
        }
        print(
            f"\n[E3 Stats: {metric}] deltas={[round(x, 5) for x in deltas]} | "
            f"mean={mean_delta:.5f}, std={std_delta:.5f}, |mean|>=2*std: {established}",
            flush=True,
        )

    final_e3_json = {
        "e2_pretrain_resnet": pretrain_resnet_records,
        "e3_depth": e3_depth,
        "e3_runs": e3_runs,
        "e3_stats": stats_out,
    }
    out_json_path = OUT_DIR / "lab1_e3_results.json"
    out_json_path.write_text(json.dumps(final_e3_json, indent=2), encoding="utf-8")
    print(f"\nSaved E3 & pre-train results to {out_json_path}", flush=True)


if __name__ == "__main__":
    main()
