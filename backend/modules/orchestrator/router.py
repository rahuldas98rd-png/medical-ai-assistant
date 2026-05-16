"""Orchestrator routes."""

import hashlib
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.core.rate_limiter import limiter
from backend.database import ConsultationHistory, get_session
from backend.modules.orchestrator.schemas.analyze import OrchestratorReport
from backend.modules.orchestrator.service import service

router = APIRouter()
log = structlog.get_logger()

ALLOWED_IMAGE_TYPES = {"chest_xray", "brain_mri", "prescription"}
MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB


def _user_key(request: Request) -> str:
    """Derive a short, non-reversible user identifier from API key or client IP."""
    api_key = request.headers.get("X-API-Key", "")
    raw = api_key if api_key else (request.client.host if request.client else "unknown")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _save_history(user_key: str, symptoms: str, report: OrchestratorReport) -> None:
    try:
        with get_session() as s:
            s.add(ConsultationHistory(
                user_key=user_key,
                symptoms_preview=symptoms[:200],
                overall_summary=report.overall_summary,
                report_json=report.model_dump(mode="json"),
            ))
    except Exception as e:
        log.error("orchestrator.history_save_failed", error=str(e))


@router.post(
    "/analyze",
    response_model=OrchestratorReport,
    summary="Multi-module health screening from symptoms + optional image",
)
@limiter.limit("15/minute")
async def analyze(
    request: Request,
    symptoms: str = Form(..., min_length=10, max_length=2000),
    age: Optional[int] = Form(None, ge=1, le=120),
    gender: Optional[str] = Form(None),
    bmi: Optional[float] = Form(None, ge=10.0, le=80.0),
    glucose: Optional[float] = Form(None, ge=30.0, le=600.0),
    systolic_bp: Optional[float] = Form(None, ge=60.0, le=260.0),
    image_type: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
) -> OrchestratorReport:
    if image_type and image_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"image_type must be one of {sorted(ALLOWED_IMAGE_TYPES)}.",
        )

    image_bytes: Optional[bytes] = None
    content_type: str = ""
    if image and image.size:
        if image.size > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 50 MB limit.")
        image_bytes = await image.read()
        content_type = image.content_type or ""

    if image_bytes and not image_type:
        raise HTTPException(
            status_code=422,
            detail="image_type is required when an image is uploaded.",
        )

    gender_clean = gender.lower().strip() if gender else None
    if gender_clean and gender_clean not in ("male", "female"):
        gender_clean = None

    report = service.analyze(
        symptoms=symptoms,
        age=age,
        gender=gender_clean,
        bmi=bmi,
        glucose=glucose,
        systolic_bp=systolic_bp,
        image_bytes=image_bytes,
        image_type=image_type,
        content_type=content_type,
    )

    _save_history(_user_key(request), symptoms, report)
    return report


@router.get(
    "/history",
    summary="Last 10 Smart Analysis consultations for the current user",
)
async def history(request: Request) -> dict:
    ukey = _user_key(request)
    try:
        with get_session() as s:
            rows = (
                s.query(ConsultationHistory)
                .filter(ConsultationHistory.user_key == ukey)
                .order_by(ConsultationHistory.created_at.desc())
                .limit(10)
                .all()
            )
        items = [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "symptoms_preview": r.symptoms_preview,
                "overall_summary": r.overall_summary,
            }
            for r in rows
        ]
    except Exception as e:
        log.error("orchestrator.history_fetch_failed", error=str(e))
        items = []

    return {"user_key": ukey, "count": len(items), "consultations": items}


@router.get("/status", summary="Orchestrator readiness — shows which sub-modules are available")
async def status() -> dict:
    from backend.modules.brain_mri.classifier import classifier as brain_classifier
    from backend.modules.manual_diagnosis.diabetes_service import service as diabetes_svc
    from backend.modules.manual_diagnosis.hypertension_service import service as htn_svc
    from backend.modules.medical_imaging.service import service as xray_svc
    from backend.modules.prescription_ocr.service import service as ocr_svc

    return {
        "sub_modules": {
            "diabetes": diabetes_svc.is_ready(),
            "hypertension": htn_svc.is_ready(),
            "chest_xray": xray_svc.is_ready(),
            "brain_mri": brain_classifier.is_ready(),
            "prescription_ocr": ocr_svc.is_ready(),
        },
        "rate_limit": "15 requests / minute per IP",
    }
