"""Experiment and run identifiers (section 5.1 of the lab spec).

EXP_ID = H(Ci, Theta_i, Di, Si)  -- deterministic, SHA-256 over a canonical
serialization of: source-code state, resolved-config hash, data + split
fingerprint, seed. Two runs sharing EXP_ID *must* be comparable.

RUN_ID = EXP_ID || N            -- EXP_ID plus a distinguisher (timestamp +
nonce). Re-running the same experiment never overwrites a previous run dir.

Hardware environment (Vi) is intentionally NOT part of EXP_ID: identical
EXP_ID guarantees identical *declared* conditions, not identical arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from datetime import datetime, timezone


def sha256_hex(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no extra whitespace, ensure_ascii."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_exp_id(
    *,
    git_commit: str,
    git_dirty: bool,
    config_hash: str,
    data_fingerprint: str,
    split_fingerprint: str,
    seed: int,
) -> str:
    """EXP_ID per eq. (0.3). All inputs are canonical strings/ints."""
    payload = canonical_json(
        {
            "Ci": {"commit": git_commit, "dirty": bool(git_dirty)},
            "Theta": config_hash,
            "Di": {"data": data_fingerprint, "split": split_fingerprint},
            "Si": int(seed),
        }
    )
    return sha256_hex(payload)


def compute_run_id(exp_id: str, *, counter: int | None = None) -> str:
    """RUN_ID = EXP_ID[:12] _ timestamp _ nonce (eq. 0.4 with N = ts||nonce).

    The full EXP_ID is preserved in manifest.json; the directory name uses a
    short prefix for readability while staying collision-resistant.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    nonce = secrets.token_hex(3)
    n = counter if counter is not None else int(time.time() * 1000) % 100000
    return f"{exp_id[:12]}_{ts}_{nonce}_{n:05d}"
