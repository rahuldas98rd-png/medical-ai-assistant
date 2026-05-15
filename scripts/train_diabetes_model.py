"""
Train the diabetes risk model.

Uses the Pima Indians Diabetes dataset — the classic benchmark for this
task. ~768 rows, 8 features. A Gradient Boosting Classifier hits ~78%
accuracy / ~0.83 ROC-AUC, which is competitive with published baselines
and trains in seconds on CPU.

Why not a neural net? On 768 rows of tabular data, gradient-boosted trees
consistently beat neural nets. The Phase 3 imaging models will use deep
learning where it shines.

Run from project root:
    python scripts/train_diabetes_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Ensure repo root is on path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import get_settings  # noqa: E402

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
DATA_PATH = Path("data/raw/pima_diabetes.csv")
COLUMN_NAMES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome",
]

# Feature columns where 0 is biologically implausible → treat as missing
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

MODEL_VERSION = "0.1.0"


# -------------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------------
def download_dataset() -> pd.DataFrame:
    """Download the Pima dataset if not already cached locally."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"[1/5] Downloading dataset to {DATA_PATH} ...")
        urlretrieve(DATA_URL, DATA_PATH)
    else:
        print(f"[1/5] Dataset already present at {DATA_PATH}, skipping download.")
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    print(f"      Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0s in clinically-impossible-to-be-zero fields with NaN for imputation."""
    print("[2/5] Cleaning data (replacing impossible zeros with NaN for imputation) ...")
    df = df.copy()
    for col in ZERO_AS_MISSING:
        n_zero = (df[col] == 0).sum()
        if n_zero > 0:
            print(f"      Column '{col}': {n_zero} zeros → NaN")
            df.loc[df[col] == 0, col] = np.nan
    return df


def build_pipeline() -> Pipeline:
    """Imputer → Gradient Boosting Classifier."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    """Cross-validated AUC + held-out test metrics."""
    print("[3/5] Cross-validating (5-fold stratified) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"      ROC-AUC (CV): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"      Hold-out accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"      Hold-out ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print("      Classification report:")
    print(classification_report(y_test, y_pred, target_names=["no_diabetes", "diabetes"]))


def fit_final(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit on ALL data after evaluation, for production use."""
    print("[4/5] Fitting final model on full dataset ...")
    pipe.fit(X, y)
    return pipe


def save_model(pipe: Pipeline, feature_names: list[str]) -> Path:
    """Bundle model + feature importances + metadata and persist."""
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.models_dir / f"diabetes_gbm_v{MODEL_VERSION}.joblib"

    gbm = pipe.named_steps["model"]
    importances = dict(zip(feature_names, gbm.feature_importances_.tolist()))

    bundle = {
        "model": pipe,
        "feature_names": feature_names,
        "feature_importances": importances,
        "version": MODEL_VERSION,
        "dataset": "Pima Indians Diabetes",
        "algorithm": "GradientBoostingClassifier",
    }
    joblib.dump(bundle, out_path)
    print(f"[5/5] Saved model bundle to: {out_path}")
    print(f"      Top features by importance:")
    for f, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"        - {f:24s} {imp:.4f}")
    return out_path


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("MediMind — training diabetes risk model")
    print("=" * 70)

    df = download_dataset()
    df = clean_data(df)

    feature_cols = COLUMN_NAMES[:-1]
    X = df[feature_cols]
    y = df["outcome"]

    pipe = build_pipeline()
    evaluate(pipe, X, y)
    pipe = fit_final(pipe, X, y)
    save_model(pipe, feature_cols)

    print("\nDone. Start the backend with: uvicorn backend.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
