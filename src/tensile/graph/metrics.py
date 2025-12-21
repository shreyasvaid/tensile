from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from tensile.graph.io import GraphEdge


@dataclass(frozen=True)
class GraphMetricsConfig:
    # betweenness can be slow; start false, turn on later
    compute_betweenness: bool = False
    # if betweenness enabled, approximate with k samples (None => exact)
    betweenness_k: int | None = 200


def _to_digraph(nodes: list[str], edges: list[GraphEdge]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from([(e.src, e.dst) for e in edges if e.type == "include"])
    return g


def compute_file_metrics(
    nodes: list[str],
    edges: list[GraphEdge],
    cfg: GraphMetricsConfig | None = None,
) -> pd.DataFrame:
    """
    Returns a dataframe with one row per file node:
      - g_in_nbrs, g_out_nbrs
      - g_pagerank
      - g_scc_size
      - g_betweenness (optional)
    """
    if cfg is None:
        cfg = GraphMetricsConfig()
    g = _to_digraph(nodes, edges)

    # Unique-neighbor degrees (stable)
    in_nbrs: dict[str, int] = {u: len(set(g.predecessors(u))) for u in g.nodes}
    out_nbrs: dict[str, int] = {u: len(set(g.successors(u))) for u in g.nodes}

    # PageRank on directed graph
    pagerank: dict[str, float] = nx.pagerank(g, alpha=0.85)

    # SCC size per node
    scc_sizes: dict[str, int] = {}
    for comp in nx.strongly_connected_components(g):
        size = len(comp)
        for u in comp:
            scc_sizes[u] = size

    data = {
        "file": list(g.nodes),
        "g_in_nbrs": [in_nbrs[u] for u in g.nodes],
        "g_out_nbrs": [out_nbrs[u] for u in g.nodes],
        "g_pagerank": [pagerank[u] for u in g.nodes],
        "g_scc_size": [scc_sizes.get(u, 1) for u in g.nodes],
    }

    df = pd.DataFrame(data)

    # Optional betweenness (can be expensive)
    if cfg.compute_betweenness:
        # For performance: compute on largest weakly connected component
        # so isolated nodes don’t dominate runtime.
        ug = g.to_undirected(as_view=True)
        largest = max(nx.connected_components(ug), key=len) if g.number_of_nodes() else set()
        sub = g.subgraph(largest).copy()

        bet = nx.betweenness_centrality(
            sub,
            k=cfg.betweenness_k,  # approx if k is set
            normalized=True,
            endpoints=False,
            seed=42,
        )
        # Fill 0 for nodes not in largest component
        df["g_betweenness"] = df["file"].map(lambda u: float(bet.get(u, 0.0)))

    return df


def write_metrics_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("g_pagerank", ascending=False).to_csv(out_path, index=False)
