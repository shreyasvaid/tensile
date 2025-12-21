from __future__ import annotations

from pathlib import Path

import pandas as pd

from tensile.graph.io import read_graph_json


def build_dataset(
    repo_root: Path,
    graph_json: Path,
    metrics_csv: Path,
    history_csv: Path,
    code_stats_csv: Path,
) -> pd.DataFrame:
    """
    Join graph metrics + git history + code stats into one row per file.
    """
    nodes, _, _ = read_graph_json(graph_json)
    base = pd.DataFrame({"file": nodes})

    metrics = pd.read_csv(metrics_csv)
    history = pd.read_csv(history_csv)
    code = pd.read_csv(code_stats_csv)

    df = base.merge(metrics, on="file", how="left")
    df = df.merge(history, on="file", how="left")
    df = df.merge(code, on="file", how="left")

    # Fill NaNs (e.g., files missing history)
    for c in df.columns:
        if c == "file":
            continue
        df[c] = df[c].fillna(0)

    return df
