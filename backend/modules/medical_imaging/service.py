"""Chest X-ray service: orchestrates classification + audit logging."""

from __future__ import annotations

import hashlib

import structlog

from backend.core.exceptions import InvalidInputError, ModelNotLoadedError
from backend.database import PredictionLog, get_session
from backend.modules.medical_imaging.chest_xray_classifier import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    MODEL_NAME,
    MODEL_VERSION,
    PATHOLOGY_DESCRIPTIONS,
    TOP_FINDING_THRESHOLD,
    classifier,
)
from backend.modules.medical_imaging.schemas.chest_xray import (
    ChestXRayResponse,
    PathologyPrediction,
)

log = structlog.get_logger()

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB — medical images can be large


def _confidence_level(prob: float) -> str:
    if prob >= CONFIDENCE_HIGH:
        return "high"
    if prob >= CONFIDENCE_LOW:
        return "moderate"
    return "low"


class ChestXRayService:
    def __init__(self) -> None:
        self.classifier = classifier

    def load(self) -> None:
        self.classifier.load()

    def is_ready(self) -> bool:
        return self.classifier.is_ready()

    def process(self, image_bytes: bytes) -> ChestXRayResponse:
        if not self.is_ready():
            raise ModelNotLoadedError(
                "Chest X-ray model is not loaded. Check backend startup logs — "
                "the model downloads ~30 MB on first run and requires internet."
            )

        if len(image_bytes) > MAX_FILE_BYTES:
            raise InvalidInputError(
                f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit."
            )

        predictions_raw, latency_ms = self.classifier.predict(image_bytes)

        predictions = [
            PathologyPrediction(
                name=p["name"],
                probability=p["probability"],
                confidence_level=_confidence_level(p["probability"]),
                description=PATHOLOGY_DESCRIPTIONS.get(p["name"], ""),
            )
            for p in predictions_raw
        ]
        top_findings = [
            p for p in predictions if p.probability >= TOP_FINDING_THRESHOLD
        ]

        self._audit_log(image_bytes, top_findings, latency_ms)

        return ChestXRayResponse(
            predictions=predictions,
            top_findings=top_findings,
            model_name=MODEL_NAME,
            image_size_processed="224x224",
            processing_time_ms=latency_ms,
        )

    def _audit_log(self, image_bytes, top_findings, latency_ms) -> None:
        try:
            with get_session() as s:
                s.add(PredictionLog(
                    module_name="medical_imaging",
                    module_version=MODEL_VERSION,
                    input_hash=hashlib.sha256(image_bytes).hexdigest(),
                    prediction={
                        "task": "chest_xray",
                        "model": MODEL_NAME,
                        "top_findings": [
                            {"name": f.name, "p": f.probability} for f in top_findings
                        ],
                    },
                    confidence=top_findings[0].probability if top_findings else None,
                    latency_ms=latency_ms,
                ))
        except Exception as e:
            log.error("chest_xray.audit_log_failed", error=str(e))


service = ChestXRayService()