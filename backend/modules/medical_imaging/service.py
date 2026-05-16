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
    ViewConfidence,
)

log = structlog.get_logger()

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB — DICOM files can be large


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

    def process(
        self,
        file_bytes: bytes,
        content_type: str = "",
        filename: str = "",
        generate_heatmaps: bool = True,
    ) -> ChestXRayResponse:
        if not self.is_ready():
            raise ModelNotLoadedError(
                "Chest X-ray model is not loaded. Check backend startup logs."
            )

        if len(file_bytes) > MAX_FILE_BYTES:
            raise InvalidInputError(
                f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit."
            )

        preds_raw, view_raw, latency_ms, input_format = self.classifier.predict(
            file_bytes,
            content_type=content_type,
            filename=filename,
            generate_heatmaps=generate_heatmaps,
        )

        predictions = [
            PathologyPrediction(
                name=p["name"],
                probability=p["probability"],
                confidence_level=_confidence_level(p["probability"]),
                description=PATHOLOGY_DESCRIPTIONS.get(p["name"], ""),
                heatmap_base64=p.get("heatmap_base64"),
            )
            for p in preds_raw
        ]
        top_findings = [
            p for p in predictions if p.probability >= TOP_FINDING_THRESHOLD
        ]

        view_confidence = ViewConfidence(**view_raw)

        self._audit_log(file_bytes, top_findings, latency_ms, input_format, view_confidence)

        return ChestXRayResponse(
            predictions=predictions,
            top_findings=top_findings,
            model_name=MODEL_NAME,
            image_size_processed="224x224",
            processing_time_ms=latency_ms,
            input_format=input_format,
            view_confidence=view_confidence,
        )

    def _audit_log(self, file_bytes, top_findings, latency_ms, input_format, view) -> None:
        try:
            with get_session() as s:
                s.add(PredictionLog(
                    module_name="medical_imaging",
                    module_version=MODEL_VERSION,
                    input_hash=hashlib.sha256(file_bytes).hexdigest(),
                    prediction={
                        "task": "chest_xray",
                        "model": MODEL_NAME,
                        "input_format": input_format,
                        "view_likely_frontal": view.likely_frontal_view,
                        "view_spread": round(view.spread, 4),
                        "top_findings": [
                            {"name": f.name, "p": round(f.probability, 4)}
                            for f in top_findings
                        ],
                    },
                    confidence=top_findings[0].probability if top_findings else None,
                    latency_ms=latency_ms,
                ))
        except Exception as e:
            log.error("chest_xray.audit_log_failed", error=str(e))


service = ChestXRayService()