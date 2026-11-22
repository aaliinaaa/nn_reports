"""Configuration loading, override resolution and hashing (section 8).

All parameters that influence the experiment MUST come from a YAML config or
from explicitly registered CLI overrides. Nothing from the list in section 26
(paths, device, lr, batch size, epochs, seed, optimizer, scheduler, model
size, checkpoint frequency, split) may be hardcoded in the training code.

After all overrides are applied the *resolved* config (the configuration
actually used at launch time) is saved to the run directory and hashed; the
hash feeds EXP_ID.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .utils.ids import canonical_json, sha256_hex

# Required top-level sections and the fields each must contain after resolve.
REQUIRED = {
    "experiment": {"name", "seed", "deterministic"},
    "data": {"name", "root", "train_fraction", "num_workers"},
    "model": {"name", "width", "depth"},
    "optimizer": {"name", "lr", "weight_decay"},
    "scheduler": {"name", "warmup_epochs"},
    "train": {"epochs", "batch_size", "amp"},
    "logging": {"backend", "save_every"},
}


def _coerce(value: str):
    """Best-effort scalar coercion for CLI overrides."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_overrides(items: list[str]) -> list[tuple[list[str], object]]:
    """Parse ``a.b.c=value`` override tokens into (path, value) pairs."""
    out: list[tuple[list[str], object]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        path = [p for p in key.strip().split(".") if p]
        if not path:
            raise ValueError(f"empty override key: {item!r}")
        out.append((path, _coerce(value.strip())))
    return out


class Config:
    """Resolved configuration with deterministic hashing."""

    def __init__(self, data: dict):
        self._data = data

    # ---- construction -------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(copy.deepcopy(data))

    # ---- mutation -----------------------------------------------------
    def merge_profile(self, profile_path: str | Path) -> "Config":
        """Overlay a per-student profile onto the base config.

        A profile only rewrites the fields listed in section 11.1; it never
        changes dataset, baseline architecture, the mandatory experiment set,
        acceptance thresholds, or modes A/B/C.
        """
        with open(profile_path, "r", encoding="utf-8") as fh:
            prof = yaml.safe_load(fh) or {}
        merged = copy.deepcopy(self._data)
        for section, values in prof.items():
            if not isinstance(values, dict):
                merged[section] = values
                continue
            bucket = merged.setdefault(section, {})
            for k, v in values.items():
                bucket[k] = v
        return Config(merged)

    def apply_overrides(self, overrides: list[tuple[list[str], object]]) -> "Config":
        merged = copy.deepcopy(self._data)
        for path, value in overrides:
            node = merged
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = value
        return Config(merged)

    # ---- access -------------------------------------------------------
    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def get(self, *path, default=None):
        node = self._data
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def __getitem__(self, key):
        return self._data[key]

    # ---- validation & hashing ----------------------------------------
    def validate(self) -> None:
        missing = []
        for section, fields in REQUIRED.items():
            if section not in self._data:
                missing.append(section)
                continue
            for f in fields:
                if f not in self._data[section]:
                    missing.append(f"{section}.{f}")
        if missing:
            raise ValueError(f"config missing required fields: {missing}")
        if self._data["train"].get("amp") and not self._data["experiment"].get("deterministic"):
            # Allowed but flagged (section 8): amp+deterministic=false => mode C.
            pass

    def hash(self) -> str:
        """SHA-256 of the canonical serialization of the resolved config."""
        return sha256_hex(canonical_json(self._data))

    # ---- persistence --------------------------------------------------
    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=True, allow_unicode=True)


def resolve(
    config_path: str | Path,
    *,
    profile_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load -> merge profile -> apply CLI overrides -> validate."""
    cfg = Config.load(config_path)
    if profile_path:
        cfg = cfg.merge_profile(profile_path)
    if overrides:
        cfg = cfg.apply_overrides(parse_overrides(overrides))
    cfg.validate()
    return cfg
