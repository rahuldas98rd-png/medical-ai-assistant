"""FastAPI routes for prescription OCR module."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.modules.prescription_ocr.schemas.prescription import PrescriptionResponse
from backend.modules.prescription_ocr.service import service

router = APIRouter()

ALLOWED_MIME = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/pdf",
}


@router.post(
    "/extract",
    response_model=PrescriptionResponse,
    summary="Extract structured info from a prescription image",
    description=(
        "Upload a prescription image (JPEG/PNG/WebP/BMP, max 10 MB). "
        "Returns structured medicines, dosages, frequencies, and instructions. "
        "**OCR can make mistakes — always verify against the original.**"
    ),
)
async def extract_prescription(
    file: UploadFile = File(..., description="Prescription image or PDF (max 10 MB)."),
) -> PrescriptionResponse:
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                f"Allowed: {sorted(ALLOWED_MIME)}"
            ),
        )
    file_bytes = await file.read()
    return service.process(file_bytes, content_type=file.content_type or "")


@router.get("/info", summary="Module status and capability info")
async def info() -> dict:
    return {
        "version": "0.1.0",
        "engines": {"tesseract": service.engine.is_available()},
        "medicine_dictionary_size": len(service._medicine_dict),
        "max_file_size_mb": 10,
        "supported_mime_types": sorted(ALLOWED_MIME),
    }