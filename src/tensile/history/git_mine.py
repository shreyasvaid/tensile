from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GitHistoryConfig:
    half_life_days: float = 30.0


def _run_git(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root)] + args,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout


def _parse_asof(asof: str) -> datetime:
    # asof is YYYY-MM-DD
    dt = datetime.strptime(asof, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _decay_weight(delta_days: float, half_life_days: float) -> float:
    return 2.0 ** (-delta_days / half_life_days)


def extract_file_history(
    repo_root: Path,
    files: Iterable[str],
    asof: str,
    cfg: GitHistoryConfig | None = None,
) -> pd.DataFrame:
    """
    Per-file git history features up to and including `asof` (YYYY-MM-DD).

    Columns:
      file
      h_commit_count
      h_author_count
      h_loc_added
      h_loc_deleted
      h_loc_churn
      h_days_since_last_change
      h_file_age_days
      h_recent_churn
      h_recent_commits
    """
    cfg = GitHistoryConfig()
    repo_root = repo_root.resolve()
    T = _parse_asof(asof)

    rows: list[dict] = []

    for f in files:
        # Commit list with author and timestamp
        try:
            log = _run_git(
                repo_root,
                [
                    "log",
                    "--follow",
                    f"--until={asof} 23:59:59",
                    "--format=%an|%at",
                    "--",
                    f,
                ],
            ).strip()
        except Exception:
            log = ""

        authors = set()
        times: list[datetime] = []

        if log:
            for line in log.splitlines():
                parts = line.split("|")
                if len(parts) != 2:
                    continue
                author, ts = parts
                authors.add(author)
                times.append(datetime.fromtimestamp(int(ts), tz=timezone.utc))

        commit_count = len(times)
        author_count = len(authors)

        # Churn + recency-weighted churn from numstat
        added_total = 0
        deleted_total = 0
        recent_churn = 0.0
        recent_commits = 0.0

        if commit_count > 0:
            try:
                numstat = _run_git(
                    repo_root,
                    [
                        "log",
                        "--follow",
                        f"--until={asof} 23:59:59",
                        "--pretty=%at",
                        "--numstat",
                        "--",
                        f,
                    ],
                ).strip()
            except Exception:
                numstat = ""

            current_commit_time: datetime | None = None
            current_w = 0.0

            for line in numstat.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Commit header timestamp
                if line.isdigit():
                    current_commit_time = datetime.fromtimestamp(int(line), tz=timezone.utc)
                    delta_days = (T - current_commit_time).total_seconds() / (3600 * 24)
                    current_w = _decay_weight(delta_days, cfg.half_life_days)
                    recent_commits += current_w
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                a, d, path = parts[0], parts[1], parts[2]

                # Skip binary files
                if a == "-" or d == "-":
                    continue

                # Only count exact path match for v1
                if path != f:
                    continue

                ai = int(a)
                di = int(d)
                added_total += ai
                deleted_total += di

                if current_commit_time is not None:
                    recent_churn += current_w * (ai + di)

        churn_total = added_total + deleted_total

        if commit_count == 0:
            days_since_last = None
            file_age_days = None
        else:
            last_time = max(times)
            first_time = min(times)
            days_since_last = (T - last_time).days
            file_age_days = (T - first_time).days

        rows.append(
            dict(
                file=f,
                h_commit_count=commit_count,
                h_author_count=author_count,
                h_loc_added=added_total,
                h_loc_deleted=deleted_total,
                h_loc_churn=churn_total,
                h_days_since_last_change=days_since_last,
                h_file_age_days=file_age_days,
                h_recent_churn=recent_churn,
                h_recent_commits=recent_commits,
            )
        )

    return pd.DataFrame(rows)
