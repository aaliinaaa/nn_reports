"""Per-student profile generator (section 11).

The profile is NOT chosen freely: it is computed from the student record-book
number via SHA-256, so it is reproducible, verifiable, and identical on every
computation. Only the fields listed in section 11.1 are derived here; the
profile never touches dataset, baseline architecture, mandatory experiments,
acceptance thresholds, or modes A/B/C.

Two different students must produce two different profiles; the same student
must always produce the same profile (test #9).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

# Ordered value sets, indexed from 0 (section 11.1).
TRAIN_FRACTIONS = [0.20, 0.25, 0.30, 0.35]
MODEL_WIDTHS = [48, 64, 96, 128]
MODEL_DEPTHS = [3, 4, 5]
BATCH_SIZES = [64, 128, 256]
OPTIMIZERS = ["adamw", "sgd"]
SCHEDULERS = ["cosine", "step"]
PROFILE_BATCH_SETS = [
    [1, 8, 32, 64],
    [1, 8, 32, 128],
    [1, 16, 64, 256],
]


def _h(student_id: str) -> str:
    return hashlib.sha256(student_id.encode("utf-8")).hexdigest()


def compute_profile(student_id: str) -> dict:
    """Return the resolved profile as a plain dict matching the spec format."""
    h = _h(student_id)

    def idx(a: int, b: int, mod: int) -> int:
        return int(h[a:b], 16) % mod

    profile = {
        "id": "P" + h[0:6],
        "student_id_hash": h,
        "base_seed": int(h[0:8], 16) % (2 ** 31),
        "profile_batch_sizes": PROFILE_BATCH_SETS[idx(20, 22, 3)],
    }
    return {
        "profile": profile,
        "data": {
            "train_fraction": TRAIN_FRACTIONS[idx(8, 10, 4)],
            "num_workers": 0,  # mode A default
        },
        "model": {
            "width": MODEL_WIDTHS[idx(10, 12, 4)],
            "depth": MODEL_DEPTHS[idx(12, 14, 3)],
        },
        "train": {
            "batch_size": BATCH_SIZES[idx(14, 16, 3)],
        },
        "optimizer": {
            "name": OPTIMIZERS[idx(16, 18, 2)],
        },
        "scheduler": {
            "name": SCHEDULERS[idx(18, 20, 2)],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute the per-student experiment profile.")
    p.add_argument("--student-id", required=True, help="record-book number, no spaces")
    p.add_argument("--out", default=None, help="write YAML profile here (e.g. configs/profiles/<ID>.yaml)")
    args = p.parse_args(argv)

    student_id = args.student_id.replace(" ", "")
    if not student_id:
        raise SystemExit("student-id is empty")

    prof = compute_profile(student_id)
    text = yaml.safe_dump(prof, sort_keys=False, allow_unicode=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"profile written: {out}")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
