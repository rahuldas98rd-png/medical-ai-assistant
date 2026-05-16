"""Service layer for liver disease risk prediction (Indian Liver Patient Dataset)."""

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
from backend.modules.manual_diagnosis.schemas.liver_disease import (
    FeatureContribution,
    LiverDiseaseRiskRequest,
    LiverDiseaseRiskResponse,
    RiskLevel,
)

log = structlog.get_logger()

FEATURE_NAMES = [
    "age",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_proteins",
    "albumin",
    "albumin_globulin_ratio",
]

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65
MODEL_VERSION = "0.1.0"

# Normal reference ranges for annotation in recommendations
NORMAL_RANGES = {
    "total_bilirubin": (0.2, 1.2),
    "direct_bilirubin": (0.0, 0.3),
    "alkaline_phosphotase": (44, 147),
    "alamine_aminotransferase": (7, 56),
    "aspartate_aminotransferase": (10, 40),
    "total_proteins": (6.3, 8.2),
    "albumin": (3.5, 5.0),
    "albumin_globulin_ratio": (1.0, 2.5),
}


class LiverDiseaseService:
    def __init__(self) -> None:
        self._model = None
        self._feature_importances: dict[str, float] | None = None
        self._model_path: Path = (
            get_settings().models_dir / "liver_disease_gbm_v0.1.0.joblib"
        )

    def load(self) -> None:
        if not self._model_path.exists():
            log.warning(
                "liver_disease.model.missing",
                path=str(self._model_path),
                hint="Run `python scripts/train_liver_disease_model.py` to create it.",
            )
            return
        bundle = joblib.load(self._model_path)
        self._model = bundle["model"]
        self._feature_importances = bundle.get("feature_importances", {})
        log.info("liver_disease.model.loaded", path=str(self._model_path))

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, req: LiverDiseaseRiskRequest) -> LiverDiseaseRiskResponse:
        if self._model is None:
            raise ModelNotLoadedError(
                f"Liver disease model not loaded. Expected at {self._model_path}. "
                "Run scripts/train_liver_disease_model.py."
            )

        start = time.perf_counter()
        x = self._to_feature_vector(req)
        proba = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        latency_ms = int((time.perf_counter() - start) * 1000)

        risk = self._score_to_risk(proba)
        contributors = self._top_contributors(req)
        recs = self._recommendations(req, risk)

        self._log_prediction(req, proba, latency_ms)

        return LiverDiseaseRiskResponse(
            risk=risk,
            top_contributors=contributors,
            recommendations=recs,
            model_version=MODEL_VERSION,
        )

    def _to_feature_vector(self, req: LiverDiseaseRiskRequest) -> np.ndarray:
        return np.array(
            [
                req.age,
                req.gender,
                req.total_bilirubin,
                req.direct_bilirubin,
                req.alkaline_phosphotase,
                req.alamine_aminotransferase,
                req.aspartate_aminotransferase,
                req.total_proteins,
                req.albumin,
                req.albumin_globulin_ratio,
            ],
            dtype=np.float64,
        )

    def _score_to_risk(self, proba: float) -> RiskLevel:
        if proba < LOW_THRESHOLD:
            return RiskLevel(
                label="low",
                score=proba,
                description=(
                    "Your liver function values suggest a low estimated risk pattern. "
                    "Maintain a liver-healthy lifestyle and routine annual check-ups."
                ),
            )
        if proba < HIGH_THRESHOLD:
            return RiskLevel(
                label="moderate",
                score=proba,
                description=(
                    "Some liver function values are outside the typical healthy range. "
                    "A review with a physician to interpret your full LFT panel is recommended."
                ),
            )
        return RiskLevel(
            label="high",
            score=proba,
            description=(
                "Several liver function values are notably elevated. Please consult a "
                "gastroenterologist or hepatologist for comprehensive liver evaluation."
            ),
        )

    def _top_contributors(self, req: LiverDiseaseRiskRequest, top_n: int = 4) -> list[FeatureContribution]:
        if not self._feature_importances:
            return []
        sorted_features = sorted(
            self._feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n]
        values: dict = {
            "age": req.age,
            "gender": req.gender,
            "total_bilirubin": req.total_bilirubin,
            "direct_bilirubin": req.direct_bilirubin,
            "alkaline_phosphotase": req.alkaline_phosphotase,
            "alamine_aminotransferase": req.alamine_aminotransferase,
            "aspartate_aminotransferase": req.aspartate_aminotransferase,
            "total_proteins": req.total_proteins,
            "albumin": req.albumin,
            "albumin_globulin_ratio": req.albumin_globulin_ratio,
        }
        return [
            FeatureContribution(feature=f, value=float(values[f]), importance=float(imp))
            for f, imp in sorted_features
        ]

    def _recommendations(self, req: LiverDiseaseRiskRequest, risk: RiskLevel) -> list[str]:
        recs: list[str] = []

        if req.total_bilirubin > NORMAL_RANGES["total_bilirubin"][1]:
            recs.append(
                f"Total bilirubin ({req.total_bilirubin} mg/dL) is above normal (<1.2). "
                "Elevated bilirubin can signal haemolysis, liver disease, or bile duct obstruction."
            )

        if req.alamine_aminotransferase > NORMAL_RANGES["alamine_aminotransferase"][1]:
            recs.append(
                f"ALT ({req.alamine_aminotransferase} IU/L) is above normal (<56). "
                "Elevated ALT is a sensitive marker of liver cell damage — avoid alcohol "
                "and hepatotoxic medications until reviewed by a doctor."
            )

        if req.aspartate_aminotransferase > NORMAL_RANGES["aspartate_aminotransferase"][1]:
            recs.append(
                f"AST ({req.aspartate_aminotransferase} IU/L) is above normal (<40). "
                "Combined AST/ALT elevation strongly suggests hepatic injury."
            )

        if req.albumin < NORMAL_RANGES["albumin"][0]:
            recs.append(
                f"Albumin ({req.albumin} g/dL) is below normal (≥3.5). "
                "Low albumin can reflect impaired liver synthetic function or poor nutrition."
            )

        if req.albumin_globulin_ratio < NORMAL_RANGES["albumin_globulin_ratio"][0]:
            recs.append(
                "Low albumin/globulin ratio may indicate chronic liver disease or "
                "autoimmune conditions — a full liver panel and specialist review is warranted."
            )

        recs.append(
            "Limit alcohol strictly — even moderate consumption accelerates liver fibrosis "
            "in the presence of pre-existing liver stress."
        )
        recs.append(
            "Maintain a healthy weight. Non-alcoholic fatty liver disease (NAFLD) affects "
            "~25% of the global population and is strongly linked to obesity."
        )
        recs.append(
            "Avoid hepatotoxic supplements (high-dose niacin, kava, certain herbal remedies). "
            "Always inform your doctor of any supplements you take."
        )

        if risk.label != "low":
            recs.insert(
                0,
                "Please consult a physician for interpretation of your liver function tests — "
                "this risk estimate does NOT replace clinical diagnosis.",
            )

        return recs

    def _log_prediction(self, req: LiverDiseaseRiskRequest, proba: float, latency_ms: int) -> None:
        from backend.core.audit_log import log_prediction
        payload = req.model_dump_json().encode("utf-8")
        log_prediction(
            module_name="manual_diagnosis",
            module_version=MODEL_VERSION,
            input_hash=hashlib.sha256(payload).hexdigest(),
            prediction={"task": "liver_disease", "probability": proba},
            confidence=proba,
            latency_ms=latency_ms,
        )


service = LiverDiseaseService()
