from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_COL = "y_bugfix_next"
ID_COL = "file"

# Columns we do NOT train on
DROP_COLS = {ID_COL, LABEL_COL}


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    feature_cols: list[str]


def train_logreg(dataset_labeled_csv: Path) -> TrainResult:
    df = pd.read_csv(dataset_labeled_csv)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].astype(float).to_numpy()
    y = df[LABEL_COL].astype(int).to_numpy()

    # Class imbalance handling: balanced weights is a strong default
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return TrainResult(model=model, feature_cols=feature_cols)


def save_model(out_path: Path, model: Pipeline, feature_cols: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, out_path)


def load_model(path: Path):
    obj = joblib.load(path)
    return obj["model"], obj["feature_cols"]
