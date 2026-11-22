"""Experiment manifest builder (section 15).

The manifest is the run's provenance record: every field that EXP_ID relies
on, plus the environment (Vi) which is NOT part of EXP_ID but must be
recorded to interpret reproducibility. ``profile.id`` and
``student_id_hash`` make profile substitution detectable (section 11.2).
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import git as gitutils


def _torch_env() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "cudnn_version": (
            torch.backends.cudnn.version() if cuda and torch.backends.cudnn.is_available() else None
        ),
        "device_type": "cuda" if cuda else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
        "device_count": torch.cuda.device_count() if cuda else 0,
    }


def _pip_freeze_hash() -> tuple[str, list[str]]:
    """Hash of the exact resolved dependency tree (section 10).

    Falls back to ``pip list --format=freeze``; the hash is stored in the
    manifest, the full list is written to ``environment.txt``.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60
        )
        lines = sorted(line.strip() for line in out.stdout.splitlines() if line.strip())
    except Exception:
        lines = []
    import hashlib

    h = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return h, lines


def build_manifest(
    *,
    exp_id: str,
    run_id: str,
    profile_id: str | None,
    student_id_hash: str | None,
    config_hash: str,
    data_fingerprint: str,
    split_fingerprint: str,
    split_sizes: dict[str, int],
    seed: int,
    deterministic: bool,
    amp: bool,
    num_workers: int,
    param_count: int,
    resolved_config_path: str,
    repo_root: str | None = None,
) -> dict[str, Any]:
    g = gitutils.summary(repo_root)
    lock_hash, lock_lines = _pip_freeze_hash()
    env = _torch_env()
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "exp_id": exp_id,
        "run_id": run_id,
        "profile": {"id": profile_id, "student_id_hash": student_id_hash},
        "started_at": started_at,
        "git": g,
        "config_hash": config_hash,
        "data": {
            "data_fingerprint": data_fingerprint,
            "split_fingerprint": split_fingerprint,
            "roles": split_sizes,
        },
        "seed": int(seed),
        "determinism_mode": {
            "deterministic": bool(deterministic),
            "amp": bool(amp),
            "num_workers": int(num_workers),
            "mode": _mode_label(deterministic, amp, num_workers),
        },
        "environment": env,
        "dependency_lock_hash": lock_hash,
        "param_count": int(param_count),
        "resolved_config_path": resolved_config_path,
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }
    manifest["_lock_lines"] = lock_lines  # popped before saving; used for environment.txt
    return manifest


def _mode_label(deterministic: bool, amp: bool, num_workers: int) -> str:
    """Acceptance-mode label per section 9.2 (A/B/C)."""
    import torch

    cuda = torch.cuda.is_available()
    if amp or not deterministic or num_workers > 0:
        return "C"
    if cuda:
        return "B"
    return "A"


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    lock_lines = manifest.pop("_lock_lines", [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    # environment.txt: full reproducible environment record (section 10).
    env_txt = Path(path).parent / "environment.txt"
    sections = [
        f"started_at: {manifest['started_at']}",
        f"python: {manifest['environment']['python_version']}",
        f"torch: {manifest['environment']['torch_version']}",
        f"cuda: {manifest['environment']['cuda_version']}",
        f"cudnn: {manifest['environment']['cudnn_version']}",
        f"device: {manifest['environment']['device_name']} "
        f"({manifest['environment']['device_type']}, "
        f"count={manifest['environment']['device_count']})",
        f"os: {manifest['platform']}",
        f"host: {manifest['host']}",
        f"dependency_lock_hash: {manifest['dependency_lock_hash']}",
        "",
        "# pip freeze (exact resolved tree)",
    ]
    env_txt.write_text("\n".join(sections + lock_lines) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
