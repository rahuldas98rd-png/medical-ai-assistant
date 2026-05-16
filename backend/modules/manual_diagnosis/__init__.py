"""
Manual Diagnosis module — risk prediction from user-entered measurements.

Phase 1: diabetes.
Phase 2: hypertension, heart disease, liver disease.
"""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.manual_diagnosis.diabetes_service import service as diabetes_service
from backend.modules.manual_diagnosis.heart_disease_service import service as heart_disease_service
from backend.modules.manual_diagnosis.hypertension_service import service as hypertension_service
from backend.modules.manual_diagnosis.liver_disease_service import service as liver_disease_service
from backend.modules.manual_diagnosis.router import router as diagnosis_router


class ManualDiagnosisModule(BaseModule):
    name = "manual_diagnosis"
    version = "0.2.0"
    description = (
        "Risk prediction for chronic conditions from user-entered clinical measurements. "
        "Covers: diabetes, hypertension, coronary heart disease, liver disease."
    )
    tags = ["diagnosis"]

    def get_router(self) -> APIRouter:
        return diagnosis_router

    def on_startup(self) -> None:
        diabetes_service.load()
        hypertension_service.load()
        heart_disease_service.load()
        liver_disease_service.load()
        self._ready = any([
            diabetes_service.is_ready(),
            hypertension_service.is_ready(),
            heart_disease_service.is_ready(),
            liver_disease_service.is_ready(),
        ])

    def health_check(self) -> dict:
        info = super().health_check()
        info["sub_models"] = {
            "diabetes": diabetes_service.is_ready(),
            "hypertension": hypertension_service.is_ready(),
            "heart_disease": heart_disease_service.is_ready(),
            "liver_disease": liver_disease_service.is_ready(),
        }
        return info
