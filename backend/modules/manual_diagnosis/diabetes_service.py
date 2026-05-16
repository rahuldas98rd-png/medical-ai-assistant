"""
Service layer for diabetes risk prediction.

Owns:
  - Loading the trained model
  - Running inference
  - Translating raw probability into a labeled risk level
  - Producing human-readable feature contributions
  - Generating evidence-based lifestyle recommendations
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog

from backend.config import get_settings
from backend.core.exceptions import ModelNotLoadedError
from backend.database import PredictionLog, get_session
from backend.modules.manual_diagnosis.schemas.diabetes import (
    DiabetesRiskRequest,
    DiabetesRiskResponse,
    FeatureContribution,
    RiskLevel,
)

log = structlog.get_logger()

# Feature order MUST match the order used during training (see scripts/train_diabetes_model.py).
FEATURE_NAMES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]

# Risk thresholds. These are illustrative — a clinical product would
# require these to be calibrated against a validation set with attention
# to base rate, false-positive cost, and false-negative cost.
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.65

MODEL_VERSION = "0.1.0"


class DiabetesService:
    """Stateless service holding the loaded model and exposing predict()."""

    def __init__(self) -> None:
        self._model = None
        self._feature_importances: dict[str, float] | None = None
        self._model_path: Path = (
            get_settings().models_dir / "diabetes_gbm_v0.1.0.joblib"
        )

    # ---- Lifecycle ----
    def load(self) -> None:
        """Load the trained model into RAM. Called once at module startup."""
        if not self._model_path.exists():
            log.warning(
                "diabetes.model.missing",
                path=str(self._model_path),
                hint="Run `python scripts/train_diabetes_model.py` to create it.",
            )
            return
        bundle = joblib.load(self._model_path)
        self._model = bundle["model"]
        self._feature_importances = bundle.get("feature_importances", {})
        log.info("diabetes.model.loaded", path=str(self._model_path))

    def is_ready(self) -> bool:
        return self._model is not None

    # ---- Inference ----
    def predict(self, req: DiabetesRiskRequest) -> DiabetesRiskResponse:
        if self._model is None:
            raise ModelNotLoadedError(
                f"Diabetes model not loaded. Expected at {self._model_path}. "
                "Run scripts/train_diabetes_model.py."
            )

        start = time.perf_counter()
        x = self._to_feature_vector(req)
        proba = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        latency_ms = int((time.perf_counter() - start) * 1000)

        risk = self._score_to_risk(proba)
        contributors = self._top_contributors(req, top_n=4)
        recs = self._recommendations(req, risk)

        response = DiabetesRiskResponse(
            risk=risk,
            top_contributors=contributors,
            recommendations=recs,
            model_version=MODEL_VERSION,
        )

        # Audit log — hash inputs, don't store raw PII
        self._log_prediction(req, proba, latency_ms)

        return response

    # ---- Helpers ----
    def _to_feature_vector(self, req: DiabetesRiskRequest) -> np.ndarray:
        return np.array(
            [
                req.pregnancies,
                req.glucose,
                req.blood_pressure,
                req.skin_thickness,
                req.insulin,
                req.bmi,
                req.diabetes_pedigree,
                req.age,
            ],
            dtype=np.float64,
        )

    def _score_to_risk(self, proba: float) -> RiskLevel:
        if proba < LOW_THRESHOLD:
            return RiskLevel(
                label="low",
                score=proba,
                description=(
                    "Your inputs suggest a low estimated risk. Maintain healthy "
                    "habits and routine check-ups."
                ),
            )
        if proba < HIGH_THRESHOLD:
            return RiskLevel(
                label="moderate",
                score=proba,
                description=(
                    "Your inputs suggest a moderate estimated risk. Consider "
                    "discussing prevention with a doctor."
                ),
            )
        return RiskLevel(
            label="high",
            score=proba,
            description=(
                "Your inputs suggest a high estimated risk. Please consult a "
                "physician promptly for proper testing (HbA1c / FPG / OGTT)."
            ),
        )

    def _top_contributors(
        self, req: DiabetesRiskRequest, top_n: int = 4
    ) -> list[FeatureContribution]:
        """
        Return the features with highest global importance. A future version
        will use SHAP for true per-prediction attributions; for now we surface
        the model's overall feature importances annotated with this request's
        values.
        """
        if not self._feature_importances:
            return []

        sorted_features = sorted(
            self._feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n]
        values = req.model_dump()
        return [
            FeatureContribution(feature=f, value=float(values[f]), importance=float(imp))
            for f, imp in sorted_features
        ]

    def _recommendations(
        self, req: DiabetesRiskRequest, risk: RiskLevel
    ) -> list[str]:
        """
        Heuristic, evidence-aligned suggestions. Not personalized medical
        advice. Sources: WHO diabetes fact sheet, ADA Standards of Care.
        """
        recs: list[str] = []

        if req.bmi >= 30:
            recs.append(
                "BMI ≥30 (obesity range). Even 5–7% weight loss can substantially "
                "lower diabetes risk."
            )
        elif req.bmi >= 25:
            recs.append("BMI in overweight range. Gradual weight loss reduces risk.")

        if req.glucose >= 140:
            recs.append(
                "Glucose reading is above the typical post-meal range. Ask your "
                "doctor about an HbA1c or oral glucose tolerance test."
            )

        if req.blood_pressure >= 90:
            recs.append(
                "Diastolic BP appears elevated. Diabetes and hypertension share "
                "risk factors — both benefit from reduced sodium and regular activity."
            )

        if req.age >= 45:
            recs.append(
                "Age ≥45 raises baseline risk; routine fasting glucose or HbA1c "
                "screening every 3 years is generally recommended."
            )

        # Always include foundational guidance
        recs.append(
            "Aim for at least 150 minutes/week of moderate physical activity "
            "(brisk walking, cycling, swimming)."
        )
        recs.append(
            "Favor whole grains, legumes, vegetables, and lean protein. Limit "
            "sugar-sweetened beverages and refined carbohydrates."
        )

        if risk.label != "low":
            recs.insert(
                0,
                "Please book a consultation with a physician for proper laboratory "
                "testing — this risk estimate does NOT replace clinical diagnosis.",
            )

        return recs

    def _log_prediction(
        self, req: DiabetesRiskRequest, proba: float, latency_ms: int
    ) -> None:
        from backend.core.audit_log import log_prediction
        payload = req.model_dump_json().encode("utf-8")
        log_prediction(
            module_name="manual_diagnosis",
            module_version=MODEL_VERSION,
            input_hash=hashlib.sha256(payload).hexdigest(),
            prediction={"task": "diabetes", "probability": proba},
            confidence=proba,
            latency_ms=latency_ms,
        )


# Module-level singleton (the module's on_startup() calls .load())
service = DiabetesService()
