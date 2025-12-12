import typer
from pathlib import Path

from tensile.paths import artifact_paths
from tensile.graph.build_includes import build_includes_graph
from tensile.graph.io import write_graph_json

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

    typer.echo(f"✔ Wrote graph: {ap.graph_json}")
    typer.echo(f"   Nodes: {meta['node_count']}, Edges: {meta['edge_count']}")
    typer.echo(
        f"   Includes: {meta['include_directives_total']}, Unresolved: {meta['unresolved_includes_total']}"
    )

# Keep the rest of your stubs below (unchanged for now)
@app.command()
def extract_history(repo: str, asof: str = typer.Option(..., help="As-of date YYYY-MM-DD")):
    raise NotImplementedError("TODO: implement extract_history")

@app.command("build-features")
def build_features(repo: str, asof: str = typer.Option(..., help="As-of date YYYY-MM-DD")):
    raise NotImplementedError("TODO: implement build_features")

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
