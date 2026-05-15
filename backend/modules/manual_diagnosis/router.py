"""FastAPI routes for the manual diagnosis module."""

from fastapi import APIRouter

from backend.modules.manual_diagnosis.diabetes_service import service as diabetes_service
from backend.modules.manual_diagnosis.schemas.diabetes import (
    DiabetesRiskRequest,
    DiabetesRiskResponse,
)

router = APIRouter()


@router.post(
    "/diabetes",
    response_model=DiabetesRiskResponse,
    summary="Estimate diabetes risk from clinical measurements",
    description=(
        "Returns a calibrated risk score (low / moderate / high) and "
        "lifestyle recommendations. **This is an educational estimate, "
        "not a diagnosis.**"
    ),
)
async def predict_diabetes(payload: DiabetesRiskRequest) -> DiabetesRiskResponse:
    return diabetes_service.predict(payload)


@router.get(
    "/conditions",
    summary="List supported conditions for this module",
)
async def list_conditions() -> dict:
    return {
        "conditions": [
            {
                "id": "diabetes",
                "name": "Type 2 Diabetes Risk",
                "status": "available" if diabetes_service.is_ready() else "model_not_loaded",
                "endpoint": "/diabetes",
            },
            # Phase 2 placeholders
            {"id": "hypertension", "name": "Hypertension Risk", "status": "coming_soon"},
            {"id": "heart_disease", "name": "Heart Disease Risk", "status": "coming_soon"},
            {"id": "liver_disease", "name": "Liver Disease Risk", "status": "coming_soon"},
        ]
    }
