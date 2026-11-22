"""Run comparator (section 20, Task 7).

Builds a summary table over several RUN_IDs, groups them by EXP_ID, and
exposes ``diff_runs(a, b)`` which prints only the differing configuration
fields. For two runs with the same EXP_ID the diff MUST be empty -- a
non-empty diff signals a canonicalization bug (section 20).

Usage:
    python -m src.compare --runs <RUN_ID_1> <RUN_ID_2> ...
    python -m src.compare --runs A B --diff
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import Config
from .diagnostics.logger import load_summary
from .diagnostics.manifest import load_manifest


def _find_run(run_id: str) -> Path:
    p = Path(run_id)
    if p.exists() and (p / "manifest.json").exists():
        return p
    return Path("runs") / run_id


def _short(h: str | None, n: int = 12) -> str:
    return (h or "")[:n]


def _row(run_dir: Path) -> dict[str, Any]:
    mf = load_manifest(run_dir / "manifest.json")
    try:
        sm = load_summary(run_dir / "summary.json")
    except FileNotFoundError:
        sm = {}
    return {
        "run_id": run_dir.name,
        "exp_id": _short(mf.get("exp_id")),
        "seed": mf.get("seed"),
        "mode": mf.get("determinism_mode", {}).get("mode"),
        "val_acc": sm.get("best_val_acc"),
        "test_acc": sm.get("test_acc"),
        "test_loss": sm.get("test_loss"),
        "params": sm.get("param_count") or mf.get("param_count"),
        "time_s": sm.get("total_train_time_s"),
        "first_batch_checksum": _short(sm.get("first_batch_checksum"), 16),
    }


def _fmt_table(rows: list[dict[str, Any]]) -> str:
    cols = ["run_id", "exp_id", "seed", "mode", "val_acc", "test_acc", "test_loss", "params", "time_s"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join(
        "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows
    )
    return "\n".join([header, sep, body])


def _group_by_exp_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["exp_id"], []).append(r)
    return groups


def diff_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """Return only the resolved-config paths that differ between two runs."""
    ca = Config.load(_find_run(str(run_a)) / "config.resolved.yaml").to_dict()
    cb = Config.load(_find_run(str(run_b)) / "config.resolved.yaml").to_dict()
    return _diff_dict(ca, cb, prefix="")


def _diff_dict(a: Any, b: Any, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            path = f"{prefix}.{k}" if prefix else k
            if k not in a:
                out[path] = {"a": None, "b": b[k]}
            elif k not in b:
                out[path] = {"a": a[k], "b": None}
            else:
                out.update(_diff_dict(a[k], b[k], path))
    elif a != b:
        out[prefix] = {"a": a, "b": b}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare several runs and group by EXP_ID.")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--diff", action="store_true", help="also diff the first two runs' configs")
    args = p.parse_args(argv)

    rows = [_row(_find_run(r)) for r in args.runs]
    print(_fmt_table(rows))
    print("\n# grouped by EXP_ID")
    for exp_id, group in _group_by_exp_id(rows).items():
        ids = [r["run_id"] for r in group]
        note = "  (same EXP_ID -> must be comparable)" if len(group) > 1 else ""
        print(f"  {exp_id}: {ids}{note}")

    if args.diff and len(args.runs) >= 2:
        d = diff_runs(args.runs[0], args.runs[1])
        print(f"\n# diff_runs({args.runs[0]}, {args.runs[1]})")
        if not d:
            print("  (empty -- configs identical, canonicalization OK)")
        else:
            for path, vals in d.items():
                print(f"  {path}: a={vals['a']!r}  b={vals['b']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
