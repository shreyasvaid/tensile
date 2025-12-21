from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report_md(
    out_path: Path,
    ranked: pd.DataFrame,
    eval_json: Path,
    meta: dict,
    top: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_data = {}
    if eval_json.exists():
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

    lines = []
    lines.append("# TENSILE Risk Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Repo: `{meta.get('repo', '')}`\n")
    lines.append(f"- As-of: `{meta.get('asof', '')}`\n")
    lines.append(f"- Horizon (days): `{meta.get('horizon_days', '')}`\n")
    lines.append(f"- Files analyzed: **{meta.get('n_files', '')}**\n")
    lines.append(
        f"- Positive rate: **{meta.get('positive_rate', 0):.3f}** ({meta.get('n_pos', '')}/{meta.get('n_files', '')})\n"
    )
    lines.append(f"- Top-K: **{top}**\n")
    lines.append(
        f"- Hit rate in top {top}: **{meta.get('hits', '')}/{top} = {meta.get('hit_rate', 0):.2f}**\n"
    )
    lines.append(f"- Lift vs base rate: **{meta.get('lift', 0):.1f}×**\n")

    if eval_data:
        lines.append("\n## Model Evaluation\n")
        p20 = eval_data.get("model", {}).get("precision_at_k", {}).get("20")
        if p20 is not None:
            lines.append(f"- Model Precision@20: **{float(p20):.3f}**\n")
        baselines = eval_data.get("baselines", {})
        if baselines:
            lines.append("- Baselines (Precision@20):\n")
            for name, b in baselines.items():
                val = b.get("precision_at_k", {}).get("20")
                if val is not None:
                    lines.append(f"  - {name}: {float(val):.3f}\n")

    lines.append("\n## Top Risky Files\n")
    lines.append("| Rank | File | Risk score | Label |\n")
    lines.append("|---:|---|---:|---:|\n")
    for i, r in enumerate(ranked.head(top).itertuples(index=False), start=1):
        lines.append(f"| {i} | `{r.file}` | {r.risk_score:.4f} | {int(r.y_bugfix_next)} |\n")

    out_path.write_text("".join(lines), encoding="utf-8")
