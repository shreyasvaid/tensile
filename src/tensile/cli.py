import typer
from pathlib import Path

from tensile.paths import artifact_paths
from tensile.graph.build_includes import build_includes_graph
from tensile.graph.io import write_graph_json
from tensile.graph.metrics import compute_file_metrics, write_metrics_csv, GraphMetricsConfig
from tensile.graph.io import read_graph_json
from tensile.history.git_mine import extract_file_history, GitHistoryConfig
from tensile.features.code_stats import compute_code_stats
from tensile.features.build_features import build_dataset




app = typer.Typer(help="TENSILE: Graph-based risk analysis for large C codebases")

@app.command("build-graph")
def build_graph(repo: str):
    """Build dependency graph artifacts for a repository (includes-only v1)."""
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        typer.echo(f"Error: repo path does not exist or is not a directory: {repo_root}")
        raise typer.Exit(code=2)


    ap = artifact_paths(Path.cwd())  # project root is current working dir (tensile repo)
    result = build_includes_graph(repo_root)

    meta = {
        "repo_root": repo_root.as_posix(),
        "node_count": len(result.nodes),
        "edge_count": len(result.edges),
        "include_edge_count": len(result.edges),
        "unresolved_includes_total": int(sum(result.unresolved_includes.values())),
        "include_directives_total": int(sum(result.include_total.values())),
    }

    write_graph_json(
        out_path=ap.graph_json,
        nodes=result.nodes,
        edges=result.edges,
        meta=meta,
    )

    typer.echo(f"✅ Wrote graph: {ap.graph_json}")
    typer.echo(f"   Nodes: {meta['node_count']}, Edges: {meta['edge_count']}")
    typer.echo(
        f"   Includes: {meta['include_directives_total']}, Unresolved: {meta['unresolved_includes_total']}"
    )

    if result.unresolved_targets:
        typer.echo("   Most common unresolved includes:")
        top_unresolved = sorted(
            result.unresolved_targets.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for name, count in top_unresolved:
            typer.echo(f"   - {name} ({count})")


    # Compute graph metrics (v1: includes-only)
    df = compute_file_metrics(result.nodes, result.edges, cfg=GraphMetricsConfig(compute_betweenness=False))
    write_metrics_csv(df, ap.metrics_csv)

    typer.echo(f"✅ Wrote metrics: {ap.metrics_csv}")
    typer.echo(f"   Top PageRank files:")
    for row in df.sort_values("g_pagerank", ascending=False).head(5).itertuples(index=False):
        typer.echo(f"   - {row.file}  (pagerank={row.g_pagerank:.6f})")


@app.command("extract-history")
def extract_history(
    repo: str,
    asof: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    half_life_days: float = typer.Option(30.0, help="Half-life (days) for recency-weighted features"),
):
    """Extract git history features as of a date."""
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        typer.echo(f"Error: repo path does not exist or is not a directory: {repo_root}")
        raise typer.Exit(code=2)

    ap = artifact_paths(Path.cwd())
    if not ap.graph_json.exists():
        typer.echo("Error: graph.json not found. Run `tensile build-graph <repo>` first.")
        raise typer.Exit(code=2)

    nodes, _, _ = read_graph_json(ap.graph_json)
    df = extract_file_history(repo_root, nodes, asof=asof, cfg=GitHistoryConfig(half_life_days=half_life_days))

    ap.history_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ap.history_csv, index=False)

    typer.echo(f"✅ Wrote history: {ap.history_csv}")
    typer.echo(f"   Rows: {len(df)}")

    top = df.sort_values("h_recent_churn", ascending=False).head(5)
    typer.echo("   Top recent-churn files:")
    for r in top.itertuples(index=False):
        typer.echo(f"   - {r.file} (recent_churn={r.h_recent_churn:.2f}, commits={r.h_commit_count})")

@app.command("build-features")
def build_features(repo: str, asof: str = typer.Option(..., help="As-of date YYYY-MM-DD")):
    """Build dataset by joining graph metrics, history, and code features."""
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        typer.echo(f"Error: repo path does not exist or is not a directory: {repo_root}")
        raise typer.Exit(code=2)

    ap = artifact_paths(Path.cwd())

    # Require upstream artifacts
    missing = [p for p in [ap.graph_json, ap.metrics_csv, ap.history_csv] if not p.exists()]
    if missing:
        typer.echo("Error: missing required artifacts. Run these first:")
        typer.echo("  tensile build-graph <repo>")
        typer.echo("  tensile extract-history <repo> --asof YYYY-MM-DD")
        raise typer.Exit(code=2)

    # Compute code stats
    nodes, _, _ = read_graph_json(ap.graph_json)
    code_df = compute_code_stats(repo_root, nodes)
    ap.code_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    code_df.to_csv(ap.code_stats_csv, index=False)
    typer.echo(f"✅ Wrote code stats: {ap.code_stats_csv}")

    # Join dataset
    df = build_dataset(
        repo_root=repo_root,
        graph_json=ap.graph_json,
        metrics_csv=ap.metrics_csv,
        history_csv=ap.history_csv,
        code_stats_csv=ap.code_stats_csv,
    )
    df.to_csv(ap.dataset_csv, index=False)
    typer.echo(f"✅ Wrote dataset: {ap.dataset_csv} (rows={len(df)}, cols={len(df.columns)})")

    # Quick peek
    top = df.sort_values("h_recent_churn", ascending=False).head(5)
    typer.echo("   Sample (top recent churn):")
    for r in top.itertuples(index=False):
        typer.echo(f"   - {r.file}")



# Stubs
@app.command()
def train(repo: str, asof: str = typer.Option(..., help="As-of date YYYY-MM-DD")):
    raise NotImplementedError("TODO: implement train")

@app.command()
def report(repo: str, top: int = typer.Option(20, help="Show top K risky files")):
    raise NotImplementedError("TODO: implement report")

@app.command()
def explain(repo: str, file: str):
    raise NotImplementedError("TODO: implement explain")

@app.command()
def analyze(repo: str, asof: str = typer.Option(..., help="As-of date YYYY-MM-DD")):
    raise NotImplementedError("TODO: implement analyze")


if __name__ == "__main__":
    app()
