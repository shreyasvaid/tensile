from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tensile.risk.train import load_model, LABEL_COL, ID_COL


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean()) if k > 0 else 0.0


def evaluate(
    dataset_labeled_csv: Path,
    model_path: Path,
    out_json: Path,
    ks: List[int] = [10, 20, 50],
) -> Dict:
    df = pd.read_csv(dataset_labeled_csv)
    y = df[LABEL_COL].astype(int).to_numpy()

    model, feat_cols = load_model(model_path)
    X = df[feat_cols].astype(float).to_numpy()
    scores = model.predict_proba(X)[:, 1]

    # Baselines
    baselines = {
        "recent_churn": df.get("h_recent_churn", pd.Series([0]*len(df))).astype(float).to_numpy(),
        "pagerank": df.get("g_pagerank", pd.Series([0]*len(df))).astype(float).to_numpy(),
        "loc": df.get("c_loc", pd.Series([0]*len(df))).astype(float).to_numpy(),
    }

    results = {
        "n_files": int(len(df)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "ks": ks,
        "model": {},
        "baselines": {},
    }

    # ROC-AUC can be unstable with very small positives but still useful
    try:
        results["model"]["roc_auc"] = float(roc_auc_score(y, scores))
    except Exception:
        results["model"]["roc_auc"] = None

    results["model"]["precision_at_k"] = {str(k): precision_at_k(y, scores, k) for k in ks}

    for name, b in baselines.items():
        try:
            auc = float(roc_auc_score(y, b))
        except Exception:
            auc = None
        results["baselines"][name] = {
            "roc_auc": auc,
            "precision_at_k": {str(k): precision_at_k(y, b, k) for k in ks},
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
