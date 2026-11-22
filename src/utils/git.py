"""Git provenance helpers.

Ci (state of source code) in the EXP_ID model is captured by:
  * commit hash
  * dirty flag (uncommitted changes present)

If the working tree is dirty, the diff is saved inside the run directory
(see ``save_git_diff``). Git history is treated as part of experiment
provenance, not as a deployment detail.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _run(args: list[str], cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def repo_root(start: str | Path | None = None) -> str:
    out = _run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    return out or str(start or Path.cwd())


def commit_hash(start: str | Path | None = None) -> str:
    """Current HEAD commit hash (full). Empty string if not a git repo."""
    return _run(["git", "rev-parse", "HEAD"], cwd=start)


def is_dirty(start: str | Path | None = None) -> bool:
    """True if the working tree has uncommitted changes."""
    out = _run(["git", "status", "--porcelain"], cwd=start)
    return bool(out)


def git_diff(start: str | Path | None = None) -> str:
    """Unified diff of uncommitted changes (staged + unstaged)."""
    return _run(["git", "diff", "HEAD"], cwd=start)


def branch(start: str | Path | None = None) -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=start)


def save_git_diff(path: str | Path, start: str | Path | None = None) -> bool:
    """Write ``git.diff`` to *path* iff the tree is dirty. Returns True if written."""
    if not is_dirty(start):
        return False
    diff = git_diff(start)
    if not diff:
        return False
    Path(path).write_text(diff, encoding="utf-8")
    return True


def summary(start: str | Path | None = None) -> dict:
    """Compact provenance record used by the manifest."""
    return {
        "commit": commit_hash(start),
        "dirty": is_dirty(start),
        "branch": branch(start),
    }
