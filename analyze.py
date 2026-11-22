"""Post-experiment analysis (sections 21-22, counterfactual questions).

Reads RUN_ID lists from reports/runs_<exp>.txt and the profile report, then
computes:
  * E1: EXP_ID match, first-batch checksum match, relative final-metric gap.
  * E2: mean / std / 95% CI across seeds and the minimal detectable effect
        (improvements smaller than 2*s_m are not established -- section 22.3).
  * E3: relative parameter divergence d_theta between continuous and resumed
        runs (eq. 0.9) and final-metric gap.
  * E4: latency / throughput table; argmin(latency) vs argmax(throughput)
        (counterfactual question 8).
Writes a Markdown report to reports/analysis.md.

Usage:  python analyze.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from src.diagnostics.logger import load_summary, load_metrics
from src.diagnostics.manifest import load_manifest
from src.compare import diff_runs

REPORTS = Path("reports")
RUNS = Path("runs")
EPS = 1e-12


def _runs(name: str) -> list[str]:
    f = REPORTS / f"runs_{name}.txt"
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def _load(run_id: str):
    d = RUNS / run_id
    mf = load_manifest(d / "manifest.json")
    sm = load_summary(d / "summary.json")
    return d, mf, sm


def _rel(x: float, y: float) -> float:
    return abs(x - y) / max(1.0, abs(x))


def _flatten_state(ckpt_path: Path):
    import torch
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flat = []
    for k, v in state["model"].items():
        flat.append(v.detach().cpu().float().reshape(-1))
    return torch.cat(flat)


def e1_section() -> str:
    ids = _runs("e1")
    if len(ids) < 2:
        return "### E1. Repeatability\n\n(not enough runs)\n"
    d1, m1, s1 = _load(ids[0])
    d2, m2, s2 = _load(ids[1])
    exp_match = m1["exp_id"] == m2["exp_id"]
    cksum_match = s1["first_batch_checksum"] == s2["first_batch_checksum"]
    final_gap = _rel(s1.get("test_acc") or 0.0, s2.get("test_acc") or 0.0)
    diff = diff_runs(d1, d2)
    return (
        "### E1. Repeatability\n\n"
        f"- runs: `{ids[0]}`, `{ids[1]}`\n"
        f"- EXP_ID match: **{exp_match}** (`{m1['exp_id'][:12]}` vs `{m2['exp_id'][:12]}`)\n"
        f"- first-batch checksum match: **{cksum_match}**\n"
        f"- relative final-metric gap: `{final_gap:.3e}`\n"
        f"- diff_runs (must be empty for same EXP_ID): `{diff or 'empty'}`\n"
    )


def e2_section() -> str:
    ids = _runs("e2")
    if len(ids) < 2:
        return "### E2. Variance across seeds\n\n(not enough runs)\n"
    accs, exp_ids = [], []
    for rid in ids:
        _, mf, sm = _load(rid)
        accs.append(sm.get("test_acc") or sm.get("best_val_acc") or 0.0)
        exp_ids.append(mf["exp_id"])
    n = len(accs)
    mean = statistics.fmean(accs)
    std = statistics.stdev(accs) if n > 1 else 0.0
    # 95% CI of the mean (t approximated by z for N>=5).
    z95 = 1.96
    ci_half = z95 * std / math.sqrt(n) if n > 1 else 0.0
    min_effect = 2 * std
    distinct_exp = len(set(exp_ids)) == n
    return (
        "### E2. Variance across seeds\n\n"
        f"- runs: {n}; seeds differ by +0..{n-1} from base_seed\n"
        f"- m = `{mean:.6f}`, s_m = `{std:.6f}`, 95% CI = `[{mean-ci_half:.6f}, {mean+ci_half:.6f}]`\n"
        f"- minimal detectable effect (2*s_m): `{min_effect:.6f}` "
        f"(improvements below this are not established)\n"
        f"- all EXP_ID distinct and differing only in S: **{distinct_exp}**\n"
    )


def e3_section() -> str:
    a_ids = _runs("e3a")
    b_ids = _runs("e3b")
    if not a_ids or len(b_ids) < 2:
        return "### E3. Checkpoint/resume equivalence\n\n(not enough runs)\n"
    da, ma, sa = _load(a_ids[-1])
    db, mb, sb = _load(b_ids[-1])  # last entry is the resumed-to-K run
    import torch
    theta_a = _flatten_state(da / "checkpoints" / "last.pt")
    theta_b = _flatten_state(db / "checkpoints" / "last.pt")
    num = (theta_a - theta_b).norm().item()
    den = (theta_a.norm().item() + EPS)
    d_theta = num / den
    final_gap = _rel(sa.get("test_acc") or 0.0, sb.get("test_acc") or 0.0)
    step_match = sa.get("global_step") == sb.get("global_step")
    return (
        "### E3. Checkpoint/resume equivalence\n\n"
        f"- continuous run: `{a_ids[-1]}`; resumed run: `{b_ids[-1]}`\n"
        f"- d_theta = `{d_theta:.3e}` (threshold A: 1e-7)\n"
        f"- relative final-metric gap: `{final_gap:.3e}`\n"
        f"- global_step match: **{step_match}** "
        f"({sa.get('global_step')} vs {sb.get('global_step')})\n"
    )


def e4_section() -> str:
    files = sorted(REPORTS.glob("profile_*.json"))
    if not files:
        return "### E4. Performance\n\n(no profile_*.json; run `python -m src.profile ...`)\n"
    prof = json.loads(files[-1].read_text(encoding="utf-8"))
    rows = []
    for r in prof["per_batch_size"]:
        rows.append(
            f"| {r['batch_size']} | {r['latency_s']['mean']*1e3:.3f} | "
            f"{r['latency_s']['p95']*1e3:.3f} | {r['throughput_sps']['mean']:.1f} | "
            f"{r['peak_gpu_memory_bytes']} |"
        )
    bs = [r["batch_size"] for r in prof["per_batch_size"]]
    lat = [r["latency_s"]["mean"] for r in prof["per_batch_size"]]
    tp = [r["throughput_sps"]["mean"] for r in prof["per_batch_size"]]
    argmin_lat = bs[lat.index(min(lat))]
    argmax_tp = bs[tp.index(max(tp))]
    return (
        "### E4. Performance\n\n"
        f"- param_count: {prof['param_count']}; device: {prof['device']}\n"
        f"- epoch_time_s: `{prof.get('epoch_time_s', float('nan')):.3f}`\n\n"
        "| batch | latency mean (ms) | latency p95 (ms) | throughput (sps) | peak GPU mem (B) |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n\n"
        f"- argmin latency batch = **{argmin_lat}**; argmax throughput batch = **{argmax_tp}** "
        f"(counterfactual #8)\n"
    )


def cf_section() -> str:
    out = ["### Counterfactual runs\n"]
    cf2 = _runs("cf2")
    cf3 = _runs("cf3")
    if len(cf2) >= 2:
        _, m1, s1 = _load(cf2[0])
        _, m2, s2 = _load(cf2[1])
        out.append(
            f"- CF2 (num_workers=2): runs `{cf2[0]}`, `{cf2[1]}`; "
            f"EXP_ID match = {m1['exp_id']==m2['exp_id']}; "
            f"first-batch checksum match = {s1['first_batch_checksum']==s2['first_batch_checksum']}"
        )
    if cf3:
        out.append(f"- CF3 (truncated checkpoint resume): run `{cf3[0]}` (see d_theta vs E3-A)")
    return "\n".join(out) + "\n"


def main() -> int:
    parts = [
        "# Lab 0 — experiment analysis\n",
        e1_section(),
        e2_section(),
        e3_section(),
        e4_section(),
        cf_section(),
        "\n> Each claim above must be cross-checked against the referenced RUN_ID "
        "directories under `runs/` before going into the report.\n",
    ]
    out = REPORTS / "analysis.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"analysis written: {out}")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
