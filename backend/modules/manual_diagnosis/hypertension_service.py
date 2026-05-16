"""Service layer for hypertension / cardiovascular risk prediction."""

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
from backend.modules.manual_diagnosis.schemas.hypertension import (
    FeatureContribution,
    HypertensionRiskRequest,
    HypertensionRiskResponse,
    RiskLevel,
)

log = structlog.get_logger()

FEATURE_NAMES = [
    "systolic_bp",
    "tobacco_kg_lifetime",
    "ldl_cholesterol",
    "adiposity",
    "family_history",
    "type_a_behavior",
    "obesity_index",
    "alcohol_units_week",
    "age",
]

LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.55
MODEL_VERSION = "0.1.0"


class HypertensionService:
    def __init__(self) -> None:
        self._model = None
        self._feature_importances: dict[str, float] | None = None
        self._model_path: Path = (
            get_settings().models_dir / "hypertension_gbm_v0.1.0.joblib"
        )

    def load(self) -> None:
        if not self._model_path.exists():
            log.warning(
                "hypertension.model.missing",
                path=str(self._model_path),
                hint="Run `python scripts/train_hypertension_model.py` to create it.",
            )
            return
        bundle = joblib.load(self._model_path)
        self._model = bundle["model"]
        self._feature_importances = bundle.get("feature_importances", {})
        log.info("hypertension.model.loaded", path=str(self._model_path))

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, req: HypertensionRiskRequest) -> HypertensionRiskResponse:
        if self._model is None:
            raise ModelNotLoadedError(
                f"Hypertension model not loaded. Expected at {self._model_path}. "
                "Run scripts/train_hypertension_model.py."
            )

        start = time.perf_counter()
        x = self._to_feature_vector(req)
        proba = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        latency_ms = int((time.perf_counter() - start) * 1000)

        risk = self._score_to_risk(proba)
        contributors = self._top_contributors(req)
        recs = self._recommendations(req, risk)

        self._log_prediction(req, proba, latency_ms)

        return HypertensionRiskResponse(
            risk=risk,
            top_contributors=contributors,
            recommendations=recs,
            model_version=MODEL_VERSION,
        )

    def _to_feature_vector(self, req: HypertensionRiskRequest) -> np.ndarray:
        return np.array(
            [
                req.systolic_bp,
                req.tobacco_kg_lifetime,
                req.ldl_cholesterol,
                req.adiposity,
                float(req.family_history),
                req.type_a_behavior,
                req.obesity_index,
                req.alcohol_units_week,
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
                    "Your inputs suggest a low estimated cardiovascular / hypertension risk. "
                    "Maintain healthy habits and annual check-ups."
                ),
            )
        if proba < HIGH_THRESHOLD:
            return RiskLevel(
                label="moderate",
                score=proba,
                description=(
                    "Your inputs suggest a moderate cardiovascular risk. Consider discussing "
                    "blood pressure monitoring and lifestyle changes with your doctor."
                ),
            )
        return RiskLevel(
            label="high",
            score=proba,
            description=(
                "Your inputs suggest a high cardiovascular risk. Please consult a physician "
                "promptly for proper blood pressure evaluation and cardiac screening."
            ),
        )

    def _top_contributors(self, req: HypertensionRiskRequest, top_n: int = 4) -> list[FeatureContribution]:
        if not self._feature_importances:
            return []
        sorted_features = sorted(
            self._feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n]
        values: dict = {
            "systolic_bp": req.systolic_bp,
            "tobacco_kg_lifetime": req.tobacco_kg_lifetime,
            "ldl_cholesterol": req.ldl_cholesterol,
            "adiposity": req.adiposity,
            "family_history": float(req.family_history),
            "type_a_behavior": req.type_a_behavior,
            "obesity_index": req.obesity_index,
            "alcohol_units_week": req.alcohol_units_week,
            "age": req.age,
        }
        return [
            FeatureContribution(feature=f, value=float(values[f]), importance=float(imp))
            for f, imp in sorted_features
        ]

    def _recommendations(self, req: HypertensionRiskRequest, risk: RiskLevel) -> list[str]:
        recs: list[str] = []

        if req.systolic_bp >= 140:
            recs.append(
                "Systolic BP ≥140 mm Hg meets the clinical threshold for hypertension. "
                "Home monitoring and a GP review are strongly advised."
            )
        elif req.systolic_bp >= 120:
            recs.append(
                "Systolic BP in the elevated range (120–139). Regular monitoring and "
                "dietary sodium reduction can prevent progression to hypertension."
            )

        if req.ldl_cholesterol >= 4.1:
            recs.append(
                "LDL cholesterol is high. Reducing saturated fats, increasing soluble fibre "
                "(oats, legumes), and regular aerobic exercise all lower LDL."
            )

        if req.tobacco_kg_lifetime > 0:
            recs.append(
                "Tobacco use is one of the strongest modifiable cardiovascular risk factors. "
                "Cessation reduces risk significantly within 1–2 years."
            )

        if req.alcohol_units_week > 14:
            recs.append(
                "Alcohol intake above 14 units/week is associated with elevated blood pressure. "
                "Reducing to ≤14 units/week (spread across the week) lowers risk."
            )

        if req.adiposity > 30 or req.obesity_index > 30:
            recs.append(
                "Elevated body fat and obesity index. Even modest weight loss (5–10%) "
                "can reduce systolic blood pressure by 5–10 mm Hg."
            )

        recs.append(
            "Aim for 30+ minutes of moderate aerobic activity (brisk walking, cycling, swimming) "
            "on most days of the week — this alone can reduce BP by 5–8 mm Hg."
        )
        recs.append(
            "Adopt a DASH-style diet: high in fruits, vegetables, whole grains, low-fat dairy; "
            "limit sodium to <2.3 g/day. Potassium-rich foods (bananas, spinach) also help."
        )

        if risk.label != "low":
            recs.insert(
                0,
                "Please consult a physician for proper blood pressure measurement and "
                "cardiovascular screening — this estimate does NOT replace clinical assessment.",
            )

        return recs

    def _log_prediction(self, req: HypertensionRiskRequest, proba: float, latency_ms: int) -> None:
        from backend.core.audit_log import log_prediction
        payload = req.model_dump_json().encode("utf-8")
        log_prediction(
            module_name="manual_diagnosis",
            module_version=MODEL_VERSION,
            input_hash=hashlib.sha256(payload).hexdigest(),
            prediction={"task": "hypertension", "probability": proba},
            confidence=proba,
            latency_ms=latency_ms,
        )


service = HypertensionService()
