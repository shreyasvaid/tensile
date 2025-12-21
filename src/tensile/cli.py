import json
from pathlib import Path

import pandas as pd
import typer

from tensile.features.build_features import build_dataset
from tensile.features.code_stats import compute_code_stats
from tensile.graph.build_includes import build_includes_graph
from tensile.graph.io import read_graph_json, write_graph_json
from tensile.graph.metrics import GraphMetricsConfig, compute_file_metrics, write_metrics_csv
from tensile.history.git_mine import GitHistoryConfig, extract_file_history
from tensile.history.labels import build_bugfix_labels
from tensile.paths import artifact_paths
from tensile.risk.evaluate import evaluate
from tensile.risk.explain import explain_file
from tensile.risk.report import rank_files
from tensile.risk.report_md import write_report_md
from tensile.risk.train import save_model, train_logreg

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
            result.unresolved_targets.items(), key=lambda x: x[1], reverse=True
        )[:10]

        for name, count in top_unresolved:
            typer.echo(f"   - {name} ({count})")

    # Compute graph metrics (v1: includes-only)
    df = compute_file_metrics(
        result.nodes, result.edges, cfg=GraphMetricsConfig(compute_betweenness=False)
    )
    write_metrics_csv(df, ap.metrics_csv)

    typer.echo(f"✅ Wrote metrics: {ap.metrics_csv}")
    typer.echo("   Top PageRank files:")
    for row in df.sort_values("g_pagerank", ascending=False).head(5).itertuples(index=False):
        typer.echo(f"   - {row.file}  (pagerank={row.g_pagerank:.6f})")


def _run_extract_history(repo_root: Path, asof: str, half_life_days: float) -> None:
    ap = artifact_paths(Path.cwd())
    if not ap.graph_json.exists():
        typer.echo("Error: graph.json not found. Run `tensile build-graph <repo>` first.")
        raise typer.Exit(code=2)

    nodes, _, _ = read_graph_json(ap.graph_json)
    df = extract_file_history(
        repo_root,
        nodes,
        asof=asof,
        cfg=GitHistoryConfig(half_life_days=half_life_days),
    )

    ap.history_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ap.history_csv, index=False)

    typer.echo(f"✅ Wrote history: {ap.history_csv}")
    typer.echo(f"   Rows: {len(df)}")

    top = df.sort_values("h_recent_churn", ascending=False).head(5)
    typer.echo("   Top recent-churn files:")
    for r in top.itertuples(index=False):
        typer.echo(
            f"   - {r.file} (recent_churn={r.h_recent_churn:.2f}, commits={r.h_commit_count})"
        )


@app.command("extract-history")
def extract_history(
    repo: str,
    asof: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    half_life_days: float = typer.Option(
        30.0, help="Half-life (days) for recency-weighted features"
    ),
):
    """Extract git history features as of a date."""
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        typer.echo(f"Error: repo path does not exist or is not a directory: {repo_root}")
        raise typer.Exit(code=2)

    _run_extract_history(repo_root, asof=asof, half_life_days=half_life_days)


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


@app.command("label")
def label(
    repo: str,
    asof: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    horizon_days: int = typer.Option(180, help="Prediction horizon in days"),
):
    """Generate supervised labels from future bug-fix commits."""
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        typer.echo(f"Error: repo path does not exist or is not a directory: {repo_root}")
        raise typer.Exit(code=2)

    ap = artifact_paths(Path.cwd())
    if not ap.graph_json.exists():
        typer.echo("Error: graph.json not found. Run `tensile build-graph <repo>` first.")
        raise typer.Exit(code=2)

    nodes, _, _ = read_graph_json(ap.graph_json)
    df = build_bugfix_labels(repo_root, nodes, asof=asof, horizon_days=horizon_days)

    ap.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ap.labels_csv, index=False)

    rate = df["y_bugfix_next"].mean() if len(df) else 0.0
    typer.echo(f"✅ Wrote labels: {ap.labels_csv}")
    typer.echo(f"   Positive rate: {rate:.3f} ({int(df['y_bugfix_next'].sum())}/{len(df)})")


@app.command("join-labels")
def join_labels():
    """Join labels into dataset.csv -> dataset_labeled.csv."""
    ap = artifact_paths(Path.cwd())

    if not ap.dataset_csv.exists():
        typer.echo(
            "Error: dataset.csv not found. Run `tensile build-features <repo> --asof ...` first."
        )
        raise typer.Exit(code=2)

    if not ap.labels_csv.exists():
        typer.echo("Error: labels.csv not found. Run `tensile label <repo> --asof ...` first.")
        raise typer.Exit(code=2)

    ds = pd.read_csv(ap.dataset_csv)
    lab = pd.read_csv(ap.labels_csv)

    out = ds.merge(lab, on="file", how="left")
    out["y_bugfix_next"] = out["y_bugfix_next"].fillna(0).astype(int)

    out.to_csv(ap.dataset_labeled_csv, index=False)
    typer.echo(
        f"✅ Wrote labeled dataset: {ap.dataset_labeled_csv} (rows={len(out)}, cols={len(out.columns)})"
    )


@app.command()
def train():
    """Train a baseline risk model on dataset_labeled.csv."""
    ap = artifact_paths(Path.cwd())
    if not ap.dataset_labeled_csv.exists():
        typer.echo("Error: dataset_labeled.csv not found. Run `tensile join-labels` first.")
        raise typer.Exit(code=2)

    tr = train_logreg(ap.dataset_labeled_csv)
    save_model(ap.model_path, tr.model, tr.feature_cols)
    typer.echo(f"✅ Saved model: {ap.model_path}")
    typer.echo(f"   Features: {len(tr.feature_cols)}")


@app.command()
def evaluate_model():
    """Evaluate model and baselines (Precision@K, ROC-AUC)."""
    ap = artifact_paths(Path.cwd())
    if not ap.dataset_labeled_csv.exists() or not ap.model_path.exists():
        typer.echo("Error: missing dataset_labeled.csv or model. Run `tensile train` first.")
        raise typer.Exit(code=2)

    res = evaluate(ap.dataset_labeled_csv, ap.model_path, ap.eval_json)
    typer.echo(f"✅ Wrote eval: {ap.eval_json}")
    typer.echo("   Precision@K (model):")
    for k, v in res["model"]["precision_at_k"].items():
        typer.echo(f"   - P@{k}: {v:.3f}")

    typer.echo("   Baselines (P@20):")
    for name, b in res["baselines"].items():
        p20 = b["precision_at_k"].get("20", 0.0)
        typer.echo(f"   - {name}: {p20:.3f}")


@app.command()
def explain(file: str, topn: int = typer.Option(8, help="Number of contributing features to show")):
    """Explain why a file is risky using model contributions."""
    ap = artifact_paths(Path.cwd())
    if not ap.dataset_labeled_csv.exists() or not ap.model_path.exists():
        typer.echo("Error: missing dataset_labeled.csv or model. Run `tensile train` first.")
        raise typer.Exit(code=2)

    ex = explain_file(ap.dataset_labeled_csv, ap.model_path, file=file, topn=topn)

    typer.echo(f"File: {ex.file}")
    typer.echo(f"Risk score: {ex.score:.4f}")
    typer.echo("Top contributors (standardized):")
    for f, v in ex.top_positive:
        typer.echo(f"  + {f}: {v:.4f}")
    typer.echo("Top negative contributors:")
    for f, v in ex.top_negative:
        typer.echo(f"  - {f}: {v:.4f}")


@app.command()
def report(
    top: int = typer.Option(20, help="Show top K risky files"),
    out_md: bool = typer.Option(False, help="Write Markdown report to data/cache/report.md"),
    repo: str = typer.Option("", help="Repository path (optional)"),
    asof: str = typer.Option("", help="As-of date YYYY-MM-DD (optional)"),
    horizon_days: int = typer.Option(0, help="Prediction horizon in days (optional)"),
    quiet: bool = typer.Option(
        False, help="Suppress printing the ranked list (still writes report.md)"
    ),
):
    """Print top risky files and optionally write report.md."""
    ap = artifact_paths(Path.cwd())
    if not ap.dataset_labeled_csv.exists() or not ap.model_path.exists():
        typer.echo("Error: missing dataset_labeled.csv or model. Run `tensile train` first.")
        raise typer.Exit(code=2)

    # 🔹 Auto-fill metadata from last analyze() run if not provided
    run_meta_path = ap.cache_dir / "run_meta.json"
    if run_meta_path.exists() and (not repo or not asof or horizon_days == 0):
        saved = json.loads(run_meta_path.read_text(encoding="utf-8"))
        repo = repo or saved.get("repo", "")
        asof = asof or saved.get("asof", "")
        horizon_days = horizon_days or int(saved.get("horizon_days", 0))

    full = rank_files(ap.dataset_labeled_csv, ap.model_path)
    ranked = full.head(top)

    total_pos = int(full["y_bugfix_next"].sum())
    n = len(full)
    hits = int(ranked["y_bugfix_next"].sum())

    typer.echo(f"Top {top} risky files:")
    typer.echo(f"Hit rate in top {top}: {hits}/{top} = {hits/top:.2f}")
    typer.echo(f"Base rate: {total_pos}/{n} = {total_pos/n:.3f}")
    lift = (hits / top) / (total_pos / n) if total_pos else 0.0
    typer.echo(f"Lift vs base rate: {lift:.1f}x")
    typer.echo("")

    if not quiet:
        for i, r in enumerate(ranked.itertuples(index=False), start=1):
            typer.echo(f"{i:>2}. {r.file}  score={r.risk_score:.4f}  label={int(r.y_bugfix_next)}")

    if out_md:
        meta = {
            "repo": repo,
            "asof": asof,
            "horizon_days": horizon_days,
            "n_files": n,
            "n_pos": total_pos,
            "positive_rate": total_pos / n if n else 0.0,
            "hits": hits,
            "hit_rate": hits / top if top else 0.0,
            "lift": lift,
        }
        write_report_md(ap.report_md, full, ap.eval_json, meta, top=top)
        typer.echo(f"📝 Wrote report: {ap.report_md}")


@app.command()
def analyze(
    repo: str,
    asof: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    horizon_days: int = typer.Option(180, help="Label horizon in days"),
    top: int = typer.Option(20, help="Top-K for report"),
    out_md: bool = typer.Option(True, help="Write report.md"),
    force: bool = typer.Option(False, help="Recompute all artifacts"),
    half_life_days: float = typer.Option(30.0, help="Half-life for git recency features"),
):
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        typer.echo(f"Error: repo path does not exist or is not a directory: {repo_root}")
        raise typer.Exit(code=2)

    ap = artifact_paths(Path.cwd())

    # 1) Graph
    if force or not ap.graph_json.exists():
        build_graph(str(repo_root))
    else:
        typer.echo(f"↪ Using cached graph: {ap.graph_json}")

    # 2) History (SLOW — biggest win)
    if force or not ap.history_csv.exists():
        _run_extract_history(repo_root, asof=asof, half_life_days=half_life_days)
    else:
        typer.echo(f"↪ Using cached history: {ap.history_csv}")

    # 3) Code features
    if force or not ap.code_stats_csv.exists() or not ap.dataset_csv.exists():
        build_features(str(repo_root), asof=asof)
    else:
        typer.echo(f"↪ Using cached features: {ap.dataset_csv}")

    # 4) Labels
    if force or not ap.labels_csv.exists():
        label(str(repo_root), asof=asof, horizon_days=horizon_days)
    else:
        typer.echo(f"↪ Using cached labels: {ap.labels_csv}")

    if force or not ap.dataset_labeled_csv.exists():
        join_labels()
    else:
        typer.echo(f"↪ Using cached labeled dataset: {ap.dataset_labeled_csv}")

    # 5) Train
    if force or not ap.model_path.exists():
        train()
    else:
        typer.echo(f"↪ Using cached model: {ap.model_path}")

    # 6) Eval
    if force or not ap.eval_json.exists():
        evaluate_model()
    else:
        typer.echo(f"↪ Using cached eval: {ap.eval_json}")

    (ap.cache_dir).mkdir(parents=True, exist_ok=True)
    ap.run_meta_json.write_text(
        json.dumps(
            {"repo": str(repo_root), "asof": asof, "horizon_days": horizon_days},
            indent=2,
        ),
        encoding="utf-8",
    )

    # 7) Report
    report(
        top=top,
        out_md=out_md,
        repo=str(repo_root),
        asof=asof,
        horizon_days=horizon_days,
        quiet=True,
    )


if __name__ == "__main__":
    app()
