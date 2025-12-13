from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd


DEFAULT_BUGFIX_RE = re.compile(
    r"(?i)\b(fix|bug|crash|segfault|overflow|leak|regression|uaf|use-after-free|null)\b"
)

def _run_git(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root)] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout

def _parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def build_bugfix_labels(
    repo_root: Path,
    files: Iterable[str],
    asof: str,
    horizon_days: int = 180,
    bugfix_regex: re.Pattern = DEFAULT_BUGFIX_RE,
) -> pd.DataFrame:
    """
    Label file u as 1 if it is modified by a bug-fix commit in (T, T+Δ], else 0.
    T = asof date (YYYY-MM-DD), Δ = horizon_days.
    """
    repo_root = repo_root.resolve()
    T = _parse_day(asof)
    end = T + timedelta(days=horizon_days)

    # We’ll query bug-fix commits in the window and collect touched file paths.
    # Format: SHA|subject
    log = _run_git(
        repo_root,
        [
            "log",
            f"--after={asof} 00:00:00",
            f"--before={end.strftime('%Y-%m-%d')} 23:59:59",
            "--format=%H|%s",
        ],
    ).strip()

    bugfix_shas: list[str] = []
    if log:
        for line in log.splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            sha, subject = parts
            if bugfix_regex.search(subject or ""):
                bugfix_shas.append(sha)

    touched: Set[str] = set()
    # For each bugfix commit, list touched files
    for sha in bugfix_shas:
        try:
            out = _run_git(repo_root, ["show", "--name-only", "--pretty=format:", sha]).strip()
        except Exception:
            continue
        for p in out.splitlines():
            p = p.strip()
            if not p:
                continue
            touched.add(p)

    files_set = set(files)
    rows = []
    for f in files:
        rows.append({"file": f, "y_bugfix_next": 1 if f in touched else 0})

    df = pd.DataFrame(rows)

    # Useful metadata for debugging
    df.attrs["bugfix_commit_count"] = len(bugfix_shas)
    df.attrs["horizon_days"] = horizon_days
    df.attrs["asof"] = asof

    return df
