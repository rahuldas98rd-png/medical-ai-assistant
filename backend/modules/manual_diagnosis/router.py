"""FastAPI routes for the manual diagnosis module."""

from fastapi import APIRouter

from backend.modules.manual_diagnosis.diabetes_service import service as diabetes_service
from backend.modules.manual_diagnosis.heart_disease_service import service as heart_disease_service
from backend.modules.manual_diagnosis.hypertension_service import service as hypertension_service
from backend.modules.manual_diagnosis.liver_disease_service import service as liver_disease_service
from backend.modules.manual_diagnosis.schemas.diabetes import (
    DiabetesRiskRequest,
    DiabetesRiskResponse,
)
from backend.modules.manual_diagnosis.schemas.heart_disease import (
    HeartDiseaseRiskRequest,
    HeartDiseaseRiskResponse,
)
from backend.modules.manual_diagnosis.schemas.hypertension import (
    HypertensionRiskRequest,
    HypertensionRiskResponse,
)
from backend.modules.manual_diagnosis.schemas.liver_disease import (
    LiverDiseaseRiskRequest,
    LiverDiseaseRiskResponse,
)

router = APIRouter()


@router.post(
    "/diabetes",
    response_model=DiabetesRiskResponse,
    summary="Estimate diabetes risk from clinical measurements",
)
async def predict_diabetes(payload: DiabetesRiskRequest) -> DiabetesRiskResponse:
    return diabetes_service.predict(payload)


@router.post(
    "/hypertension",
    response_model=HypertensionRiskResponse,
    summary="Estimate hypertension / cardiovascular risk from clinical measurements",
)
async def predict_hypertension(payload: HypertensionRiskRequest) -> HypertensionRiskResponse:
    return hypertension_service.predict(payload)


@router.post(
    "/heart_disease",
    response_model=HeartDiseaseRiskResponse,
    summary="Estimate coronary heart disease risk from clinical measurements",
)
async def predict_heart_disease(payload: HeartDiseaseRiskRequest) -> HeartDiseaseRiskResponse:
    return heart_disease_service.predict(payload)


@router.post(
    "/liver_disease",
    response_model=LiverDiseaseRiskResponse,
    summary="Estimate liver disease risk from liver function test values",
)
async def predict_liver_disease(payload: LiverDiseaseRiskRequest) -> LiverDiseaseRiskResponse:
    return liver_disease_service.predict(payload)


@router.get("/conditions", summary="List supported conditions and their readiness status")
async def list_conditions() -> dict:
    return {
        "conditions": [
            {
                "id": "diabetes",
                "name": "Type 2 Diabetes Risk",
                "status": "available" if diabetes_service.is_ready() else "model_not_loaded",
                "endpoint": "/diabetes",
                "train_script": "scripts/train_diabetes_model.py",
            },
            {
                "id": "hypertension",
                "name": "Hypertension / Cardiovascular Risk",
                "status": "available" if hypertension_service.is_ready() else "model_not_loaded",
                "endpoint": "/hypertension",
                "train_script": "scripts/train_hypertension_model.py",
            },
            {
                "id": "heart_disease",
                "name": "Coronary Heart Disease Risk",
                "status": "available" if heart_disease_service.is_ready() else "model_not_loaded",
                "endpoint": "/heart_disease",
                "train_script": "scripts/train_heart_disease_model.py",
            },
            {
                "id": "liver_disease",
                "name": "Liver Disease Risk",
                "status": "available" if liver_disease_service.is_ready() else "model_not_loaded",
                "endpoint": "/liver_disease",
                "train_script": "scripts/train_liver_disease_model.py",
            },
        ]
    }
