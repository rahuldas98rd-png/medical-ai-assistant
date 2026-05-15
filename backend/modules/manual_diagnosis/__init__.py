"""
Manual Diagnosis module — risk prediction from user-entered measurements.

Phase 1: diabetes.
Phase 2 will add hypertension, heart disease, liver disease.
"""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.manual_diagnosis.diabetes_service import service as diabetes_service
from backend.modules.manual_diagnosis.router import router as diagnosis_router


class ManualDiagnosisModule(BaseModule):
    name = "manual_diagnosis"
    version = "0.1.0"
    description = (
        "Risk prediction for chronic conditions (currently: diabetes) "
        "from user-entered clinical measurements."
    )
    tags = ["diagnosis"]

    def get_router(self) -> APIRouter:
        return diagnosis_router

    def on_startup(self) -> None:
        diabetes_service.load()
        self._ready = diabetes_service.is_ready()

    def health_check(self) -> dict:
        info = super().health_check()
        info["sub_models"] = {"diabetes": diabetes_service.is_ready()}
        return info
