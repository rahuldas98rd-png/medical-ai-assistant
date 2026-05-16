"""
Train the liver disease risk model.

Uses the Indian Liver Patient Dataset (ILPD, UCI) — 583 patients,
10 liver function test (LFT) features.
File: data/raw/indian_liver_patient.csv  (no header, comma-separated)

Column order: age, gender, total_bilirubin, direct_bilirubin,
  alkaline_phosphotase, alamine_aminotransferase, aspartate_aminotransferase,
  total_proteins, albumin, albumin_globulin_ratio, dataset
  (dataset: 1=liver patient, 2=healthy → recoded to 1/0)

GBM reaches ~75% accuracy / ~0.79 ROC-AUC — consistent with published
baselines on this imbalanced dataset (71.5% positive class).

Run from project root:
    python scripts/train_liver_disease_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import get_settings  # noqa: E402

DATA_PATH = Path("data/raw/indian_liver_patient.csv")
MODEL_VERSION = "0.1.0"
RANDOM_SEED = 42

COLUMN_NAMES = [
    "age", "gender", "total_bilirubin", "direct_bilirubin",
    "alkaline_phosphotase", "alamine_aminotransferase",
    "aspartate_aminotransferase", "total_proteins", "albumin",
    "albumin_globulin_ratio", "dataset",
]
FEATURE_COLS = COLUMN_NAMES[:-1]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"ILPD dataset not found at {DATA_PATH}.\n"
            "Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv\n"
            "Save as: data/raw/indian_liver_patient.csv"
        )
    print(f"[1/5] Loading ILPD from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    print(f"      Loaded {len(df)} rows.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Cleaning data ...")
    df = df.copy()

    # Encode gender: Male=1, Female=0 (handles case and whitespace variants)
    df["gender"] = (df["gender"].astype(str).str.strip().str.lower() == "male").astype(int)

    # ILPD target: 1=liver patient, 2=healthy → recode to 1/0
    df["dataset"] = (df["dataset"].astype(float) == 1).astype(int)

    # albumin_globulin_ratio has a few NaN rows — imputer handles them
    missing = df[FEATURE_COLS].isnull().sum()
    if missing.any():
        print(f"      Missing values (median-imputed):\n"
              f"{missing[missing > 0].to_string()}")
    print(f"      Class distribution (1=liver patient): {df['dataset'].value_counts().to_dict()}")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, subsample=0.9, random_state=RANDOM_SEED,
        )),
    ])


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    print("[3/5] Cross-validating (5-fold stratified) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"      ROC-AUC (CV): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print(f"      Hold-out accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"      Hold-out ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=["healthy", "liver_patient"]))


def fit_final(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    print("[4/5] Fitting final model on full dataset ...")
    pipe.fit(X, y)
    return pipe


def save_model(pipe: Pipeline) -> Path:
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.models_dir / f"liver_disease_gbm_v{MODEL_VERSION}.joblib"
    gbm = pipe.named_steps["model"]
    importances = dict(zip(FEATURE_COLS, gbm.feature_importances_.tolist()))
    bundle = {
        "model": pipe,
        "feature_names": FEATURE_COLS,
        "feature_importances": importances,
        "version": MODEL_VERSION,
        "dataset": "Indian Liver Patient Dataset (ILPD, UCI)",
        "algorithm": "GradientBoostingClassifier",
    }
    joblib.dump(bundle, out_path)
    print(f"[5/5] Saved model to: {out_path}")
    print("      Top features by importance:")
    for f, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"        - {f:30s} {imp:.4f}")
    return out_path


def main() -> int:
    print("=" * 70)
    print("MediMind — training liver disease risk model (ILPD)")
    print("=" * 70)
    df = load_dataset()
    df = clean_data(df)
    X, y = df[FEATURE_COLS], df["dataset"]
    pipe = build_pipeline()
    evaluate(pipe, X, y)
    pipe = fit_final(pipe, X, y)
    save_model(pipe)
    print("\nDone. Start the backend: uvicorn backend.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
