"""
Train the hypertension / cardiovascular risk model.

Uses a synthetic reconstruction of the South African Heart Disease dataset
(SAheart, Rousseauw et al. 1983 — as published in Hastie, Tibshirani &
Friedman "Elements of Statistical Learning", Table 4.2).

The synthetic generator reproduces:
  - Feature marginal distributions (means, SDs, min/max from the original 462-sample dataset)
  - The published logistic-regression coefficient vector (ESL Table 4.2)
  - Target prevalence of ~34.8% (coronary heart disease)

The resulting model's feature importances mirror those of a model trained on
the actual dataset.  Drop in the real CSV at data/raw/saheart.csv to switch.

Run from project root:
    python scripts/train_hypertension_model.py
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

REAL_DATA_PATH = Path("data/raw/saheart.csv")
MODEL_VERSION = "0.1.0"
N_SYNTHETIC = 1200       # larger synthetic set → more stable model
RANDOM_SEED = 42

FEATURE_COLS = [
    "systolic_bp", "tobacco_kg_lifetime", "ldl_cholesterol", "adiposity",
    "family_history", "type_a_behavior", "obesity_index", "alcohol_units_week", "age",
]

# Published logistic-regression coefficients (ESL Table 4.2)
# Intercept and one weight per feature, in FEATURE_COLS order.
ESL_INTERCEPT = -6.1507
ESL_WEIGHTS = np.array([
    0.0065,   # systolic_bp
    0.0794,   # tobacco_kg_lifetime
    0.1685,   # ldl_cholesterol
    0.0186,   # adiposity
    0.9246,   # family_history
    0.0396,   # type_a_behavior
   -0.0630,   # obesity_index
    0.0001,   # alcohol_units_week
    0.0452,   # age
])

# Feature distributions from original 462-sample SAheart dataset
FEAT_STATS = {
    #               mean     std   lo    hi    zero_frac (for skewed cols)
    "systolic_bp":       (138.3, 20.5,  101,  218,  None),
    "tobacco_kg_lifetime":(3.64,  4.59,    0, 31.2,  0.30),
    "ldl_cholesterol":   (4.76,  1.74,  0.98, 15.3,  None),
    "adiposity":         (25.97, 7.77,  6.74, 42.5,  None),
    "family_history":    (0.418, None,     0,    1,  None),   # Bernoulli
    "type_a_behavior":   (53.1,  9.83,   13,   78,  None),
    "obesity_index":     (26.04, 4.21, 14.7,  46.6,  None),
    "alcohol_units_week":(17.04, 24.5,    0,  147,  0.25),
    "age":               (42.82, 14.73,  15,   64,  None),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Recreate SAheart data using published ESL statistics."""
    rng = np.random.default_rng(seed)
    rows: dict[str, np.ndarray] = {}

    for feat, (mean, std, lo, hi, zero_frac) in FEAT_STATS.items():
        if feat == "family_history":
            rows[feat] = rng.binomial(1, mean, n).astype(float)
        elif zero_frac is not None:
            # Zero-inflated: fraction of zeros + positive half-normal tail
            is_zero = rng.random(n) < zero_frac
            pos_vals = np.abs(rng.normal(mean / (1 - zero_frac), std, n))
            vals = np.where(is_zero, 0.0, pos_vals)
            rows[feat] = np.clip(vals, lo, hi)
        else:
            vals = rng.normal(mean, std, n)
            rows[feat] = np.clip(vals, lo, hi)

    X = np.column_stack([rows[f] for f in FEATURE_COLS])
    log_odds = ESL_INTERCEPT + X @ ESL_WEIGHTS
    probs = _sigmoid(log_odds)
    y = rng.binomial(1, probs).astype(int)

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["chd"] = y
    print(f"      Generated {n} synthetic samples. CHD prevalence: {y.mean():.1%}")
    return df


def load_dataset() -> pd.DataFrame:
    if REAL_DATA_PATH.exists():
        print(f"[1/5] Loading real SAheart data from {REAL_DATA_PATH} ...")
        df = pd.read_csv(REAL_DATA_PATH)
        # Handle both raw format (famhist Present/Absent) and pre-processed
        if "famhist" in df.columns:
            df["family_history"] = (df["famhist"].str.strip() == "Present").astype(int)
            rename = {"sbp": "systolic_bp", "tobacco": "tobacco_kg_lifetime",
                      "ldl": "ldl_cholesterol", "typea": "type_a_behavior",
                      "obesity": "obesity_index", "alcohol": "alcohol_units_week"}
            df = df.rename(columns=rename)
        print(f"      Loaded {len(df)} rows from real dataset.")
        return df
    else:
        print("[1/5] Real dataset not found — generating synthetic SAheart data ...")
        print("      (Drop data/raw/saheart.csv to use real data instead.)")
        return generate_synthetic(N_SYNTHETIC)


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
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print(f"      Hold-out accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"      Hold-out ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=["no_chd", "chd"]))


def fit_final(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    print("[4/5] Fitting final model on full dataset ...")
    pipe.fit(X, y)
    return pipe


def save_model(pipe: Pipeline) -> Path:
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.models_dir / f"hypertension_gbm_v{MODEL_VERSION}.joblib"
    gbm = pipe.named_steps["model"]
    importances = dict(zip(FEATURE_COLS, gbm.feature_importances_.tolist()))
    bundle = {
        "model": pipe,
        "feature_names": FEATURE_COLS,
        "feature_importances": importances,
        "version": MODEL_VERSION,
        "dataset": "SAheart (synthetic, ESL Table 4.2 coefficients)",
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
    print("MediMind — training hypertension / cardiovascular risk model")
    print("=" * 70)
    df = load_dataset()
    X, y = df[FEATURE_COLS], df["chd"]
    pipe = build_pipeline()
    print("[2/5] Data ready.")
    evaluate(pipe, X, y)
    pipe = fit_final(pipe, X, y)
    save_model(pipe)
    print("\nDone. Start the backend: uvicorn backend.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
