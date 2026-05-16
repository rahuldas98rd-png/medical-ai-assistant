"""Service layer for heart disease risk prediction (Cleveland dataset, XGBoost)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import structlog

from backend.config import get_settings
from backend.core.exceptions import ModelNotLoadedError
from backend.database import PredictionLog, get_session
from backend.modules.manual_diagnosis.schemas.heart_disease import (
    FeatureContribution,
    HeartDiseaseRiskRequest,
    HeartDiseaseRiskResponse,
    RiskLevel,
)

log = structlog.get_logger()

FEATURE_NAMES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar_gt120",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "st_depression",
    "st_slope",
    "num_major_vessels",
    "thalassemia",
]

LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.55
MODEL_VERSION = "0.1.0"


class HeartDiseaseService:
    def __init__(self) -> None:
        self._model = None
        self._feature_importances: dict[str, float] | None = None
        self._model_path: Path = (
            get_settings().models_dir / "heart_disease_xgb_v0.1.0.joblib"
        )

    def load(self) -> None:
        if not self._model_path.exists():
            log.warning(
                "heart_disease.model.missing",
                path=str(self._model_path),
                hint="Run `python scripts/train_heart_disease_model.py` to create it.",
            )
            return
        bundle = joblib.load(self._model_path)
        self._model = bundle["model"]
        self._feature_importances = bundle.get("feature_importances", {})
        log.info("heart_disease.model.loaded", path=str(self._model_path))

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, req: HeartDiseaseRiskRequest) -> HeartDiseaseRiskResponse:
        if self._model is None:
            raise ModelNotLoadedError(
                f"Heart disease model not loaded. Expected at {self._model_path}. "
                "Run scripts/train_heart_disease_model.py."
            )

        start = time.perf_counter()
        x = self._to_feature_vector(req)
        proba = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        latency_ms = int((time.perf_counter() - start) * 1000)

        risk = self._score_to_risk(proba)
        contributors = self._top_contributors(req)
        recs = self._recommendations(req, risk)

        self._log_prediction(req, proba, latency_ms)

        return HeartDiseaseRiskResponse(
            risk=risk,
            top_contributors=contributors,
            recommendations=recs,
            model_version=MODEL_VERSION,
        )

    def _to_feature_vector(self, req: HeartDiseaseRiskRequest) -> np.ndarray:
        return np.array(
            [
                req.age,
                req.sex,
                req.chest_pain_type,
                req.resting_bp,
                req.cholesterol,
                req.fasting_blood_sugar_gt120,
                req.resting_ecg,
                req.max_heart_rate,
                req.exercise_angina,
                req.st_depression,
                req.st_slope,
                req.num_major_vessels,
                req.thalassemia,
            ],
            dtype=np.float64,
        )

    def _score_to_risk(self, proba: float) -> RiskLevel:
        if proba < LOW_THRESHOLD:
            return RiskLevel(
                label="low",
                score=proba,
                description=(
                    "Your inputs suggest a low estimated heart disease risk. "
                    "Continue regular check-ups and a heart-healthy lifestyle."
                ),
            )
        if proba < HIGH_THRESHOLD:
            return RiskLevel(
                label="moderate",
                score=proba,
                description=(
                    "Your inputs suggest moderate heart disease risk. Discuss cholesterol, "
                    "blood pressure management, and lifestyle changes with your doctor."
                ),
            )
        return RiskLevel(
            label="high",
            score=proba,
            description=(
                "Your inputs suggest high heart disease risk. Please consult a cardiologist "
                "promptly — further tests (ECG, stress test, lipid panel) are warranted."
            ),
        )

    def _top_contributors(self, req: HeartDiseaseRiskRequest, top_n: int = 4) -> list[FeatureContribution]:
        if not self._feature_importances:
            return []
        sorted_features = sorted(
            self._feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n]
        values: dict = {
            "age": req.age,
            "sex": req.sex,
            "chest_pain_type": req.chest_pain_type,
            "resting_bp": req.resting_bp,
            "cholesterol": req.cholesterol,
            "fasting_blood_sugar_gt120": req.fasting_blood_sugar_gt120,
            "resting_ecg": req.resting_ecg,
            "max_heart_rate": req.max_heart_rate,
            "exercise_angina": req.exercise_angina,
            "st_depression": req.st_depression,
            "st_slope": req.st_slope,
            "num_major_vessels": req.num_major_vessels,
            "thalassemia": req.thalassemia,
        }
        return [
            FeatureContribution(feature=f, value=float(values[f]), importance=float(imp))
            for f, imp in sorted_features
        ]

    def _recommendations(self, req: HeartDiseaseRiskRequest, risk: RiskLevel) -> list[str]:
        recs: list[str] = []

        if req.chest_pain_type == 0:
            recs.append(
                "Typical angina (chest pain with exertion, relieved by rest) is a direct "
                "indicator of coronary artery disease — please see a cardiologist promptly."
            )

        if req.cholesterol >= 240:
            recs.append(
                "Total cholesterol ≥240 mg/dL is high. Dietary changes (less saturated fat, "
                "more fibre) and statin therapy (if prescribed) can significantly reduce risk."
            )
        elif req.cholesterol >= 200:
            recs.append("Borderline cholesterol (200–239 mg/dL). Monitor with a lipid panel annually.")

        if req.resting_bp >= 140:
            recs.append(
                "Elevated resting BP (≥140 mm Hg). Sustained hypertension doubles heart disease "
                "risk — a GP review is recommended."
            )

        if req.exercise_angina == 1:
            recs.append(
                "Exercise-induced angina significantly raises the likelihood of coronary artery "
                "disease. An exercise stress ECG or imaging study is warranted."
            )

        if req.num_major_vessels > 0:
            recs.append(
                "Blocked major vessels found on fluoroscopy (if this was recorded from a prior "
                "test) indicate established coronary artery disease — maintain cardiology follow-up."
            )

        recs.append(
            "Heart-healthy diet: Mediterranean-style eating (olive oil, fish, legumes, "
            "vegetables, nuts) reduces cardiovascular events by ~30%."
        )
        recs.append(
            "150 min/week moderate aerobic activity (brisk walking, cycling) improves "
            "lipid profiles, lowers BP, and reduces cardiac event risk independently."
        )

        if risk.label != "low":
            recs.insert(
                0,
                "Please consult a cardiologist for proper cardiac evaluation — this estimate "
                "does NOT replace clinical diagnosis.",
            )

        return recs

    def _log_prediction(self, req: HeartDiseaseRiskRequest, proba: float, latency_ms: int) -> None:
        from backend.core.audit_log import log_prediction
        payload = req.model_dump_json().encode("utf-8")
        log_prediction(
            module_name="manual_diagnosis",
            module_version=MODEL_VERSION,
            input_hash=hashlib.sha256(payload).hexdigest(),
            prediction={"task": "heart_disease", "probability": proba},
            confidence=proba,
            latency_ms=latency_ms,
        )


service = HeartDiseaseService()
