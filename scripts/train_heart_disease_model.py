"""
Train the heart disease risk model.

Uses the Cleveland Heart Disease dataset (UCI) — 303 patients, 13 features.
File: data/raw/heart_disease_cleveland.csv  (no header, comma-separated)

Original Cleveland encodings (remapped to 0-indexed for the API):
  cp:    1=typical angina → 0, 2=atypical → 1, 3=non-anginal → 2, 4=asymptomatic → 3
  slope: 1=upsloping → 0, 2=flat → 1, 3=downsloping → 2
  thal:  3=normal → 0, 6=fixed defect → 1, 7=reversible defect → 2
  target: 0=no disease, 1-4=disease → binarised to 0/1

XGBoost reaches ~85% accuracy / ~0.91 ROC-AUC on this dataset.

Run from project root:
    python scripts/train_heart_disease_model.py
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

DATA_PATH = Path("data/raw/heart_disease_cleveland.csv")
MODEL_VERSION = "0.1.0"
RANDOM_SEED = 42

COLUMN_NAMES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar_gt120", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope",
    "num_major_vessels", "thalassemia", "target",
]
FEATURE_COLS = COLUMN_NAMES[:-1]

# Remap Cleveland-original values to 0-indexed (to match the API schema)
CP_MAP    = {1: 0, 2: 1, 3: 2, 4: 3}
SLOPE_MAP = {1: 0, 2: 1, 3: 2}
THAL_MAP  = {3: 0, 6: 1, 7: 2}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleveland dataset not found at {DATA_PATH}.\n"
            "Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data\n"
            "Save as: data/raw/heart_disease_cleveland.csv"
        )
    print(f"[1/5] Loading Cleveland Heart Disease dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    print(f"      Loaded {len(df)} rows.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Cleaning and remapping encodings ...")
    df = df.copy()

    # Replace '?' strings with NaN; convert all columns to numeric
    df.replace("?", np.nan, inplace=True)
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remap Cleveland-original encodings to 0-indexed (matches API schema)
    df["chest_pain_type"]  = df["chest_pain_type"].map(CP_MAP)
    df["st_slope"]         = df["st_slope"].map(SLOPE_MAP)
    df["thalassemia"]      = df["thalassemia"].map(THAL_MAP)

    # Binarise target: 0=no disease, 1=disease (any grade 1-4)
    df["target"] = (df["target"] > 0).astype(int)

    missing = df[FEATURE_COLS].isnull().sum()
    if missing.any():
        print(f"      Missing values (median-imputed):\n"
              f"{missing[missing > 0].to_string()}")
    print(f"      Class distribution: {df['target'].value_counts().to_dict()}")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_SEED,
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
    print(classification_report(y_test, y_pred, target_names=["no_disease", "disease"]))


def fit_final(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    print("[4/5] Fitting final model on full dataset ...")
    pipe.fit(X, y)
    return pipe


def save_model(pipe: Pipeline) -> Path:
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.models_dir / f"heart_disease_xgb_v{MODEL_VERSION}.joblib"
    gbm = pipe.named_steps["model"]
    importances = dict(zip(FEATURE_COLS, gbm.feature_importances_.tolist()))
    bundle = {
        "model": pipe,
        "feature_names": FEATURE_COLS,
        "feature_importances": importances,
        "version": MODEL_VERSION,
        "dataset": "Cleveland Heart Disease (UCI)",
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
    print("MediMind — training heart disease risk model (HistGBM / Cleveland)")
    print("=" * 70)
    df = load_dataset()
    df = clean_data(df)
    X, y = df[FEATURE_COLS], df["target"]
    pipe = build_pipeline()
    evaluate(pipe, X, y)
    pipe = fit_final(pipe, X, y)
    save_model(pipe)
    print("\nDone. Start the backend: uvicorn backend.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
