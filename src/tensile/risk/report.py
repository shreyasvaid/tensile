from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from tensile.risk.train import load_model, LABEL_COL, ID_COL

def rank_files(dataset_labeled_csv: Path, model_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_labeled_csv)
    model, feat_cols = load_model(model_path)

    X = df[feat_cols].astype(float).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["risk_score"] = scores
    df = df.sort_values("risk_score", ascending=False)
    return df[[ID_COL, "risk_score", LABEL_COL]]
