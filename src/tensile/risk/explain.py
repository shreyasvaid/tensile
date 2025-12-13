from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from tensile.risk.train import load_model, ID_COL

@dataclass(frozen=True)
class Explanation:
    file: str
    score: float
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]

def explain_file(
    dataset_labeled_csv: Path,
    model_path: Path,
    file: str,
    topn: int = 8,
) -> Explanation:
    df = pd.read_csv(dataset_labeled_csv)
    if file not in set(df[ID_COL]):
        raise ValueError(f"File not found in dataset: {file}")

    model, feat_cols = load_model(model_path)

    row = df[df[ID_COL] == file].iloc[0]
    x = row[feat_cols].astype(float).to_numpy().reshape(1, -1)

    # Score
    score = float(model.predict_proba(x)[0, 1])

    # Extract coefficients from pipeline
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    coefs = clf.coef_.reshape(-1)

    # Contribution in standardized space: coef * z
    z = scaler.transform(x).reshape(-1)
    contrib = coefs * z

    pairs = list(zip(feat_cols, contrib))
    pairs_sorted = sorted(pairs, key=lambda t: t[1], reverse=True)

    top_pos = [(f, float(v)) for f, v in pairs_sorted[:topn]]
    top_neg = [(f, float(v)) for f, v in sorted(pairs, key=lambda t: t[1])[:topn]]

    return Explanation(file=file, score=score, top_positive=top_pos, top_negative=top_neg)
