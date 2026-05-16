"""Routes for medical imaging module."""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.modules.medical_imaging.schemas.chest_xray import ChestXRayResponse
from backend.modules.medical_imaging.service import service

router = APIRouter()

ALLOWED_MIME = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
    "application/dicom", "application/x-dicom",
    "application/octet-stream",  # DICOMs sometimes upload with this generic type
}


@router.post(
    "/chest_xray",
    response_model=ChestXRayResponse,
    summary="Classify a chest X-ray (image or DICOM) for 18 pathologies",
)
async def classify_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray: PNG/JPEG/DICOM, max 50 MB."),
    heatmaps: bool = Query(default=True, description="Generate Grad-CAM heatmaps for top findings."),
) -> ChestXRayResponse:
    if file.content_type not in ALLOWED_MIME:
        # Allow if extension looks like DICOM (octet-stream fallback above
        # should cover most cases, but be permissive)
        if not (file.filename or "").lower().endswith((".dcm", ".dicom")):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported content type: {file.content_type}. "
                    f"Allowed: {sorted(ALLOWED_MIME)}"
                ),
            )
    file_bytes = await file.read()
    return service.process(
        file_bytes,
        content_type=file.content_type or "",
        filename=file.filename or "",
        generate_heatmaps=heatmaps,
    )


@router.get("/info", summary="Module info & status")
async def info() -> dict:
    return {
        "version": "0.2.0",
        "tasks": ["chest_xray"],
        "features": ["dicom_input", "grad_cam_heatmaps", "view_confidence"],
        "model_ready": service.is_ready(),
        "max_file_size_mb": 50,
        "supported_mime_types": sorted(ALLOWED_MIME),
    }