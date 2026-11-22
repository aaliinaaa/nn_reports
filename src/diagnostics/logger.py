"""Machine-readable metric logging (section 18).

The local minimum log is JSONL and is always written, even if an external
backend (TB/MLflow/W&B) is added later. Records one line per epoch plus the
final summary. Required fields: train loss, val loss, train metric, val
metric, learning rate, epoch, global step, epoch time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metrics(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
