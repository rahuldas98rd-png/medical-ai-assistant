"""Routes for medical imaging module."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.modules.medical_imaging.schemas.chest_xray import ChestXRayResponse
from backend.modules.medical_imaging.service import service

router = APIRouter()

ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}


@router.post(
    "/chest_xray",
    response_model=ChestXRayResponse,
    summary="Classify a chest X-ray for 18 pathologies",
    description=(
        "Multi-label classification using DenseNet-121 (TorchXRayVision), "
        "pretrained on union of NIH/CheXpert/MIMIC-CXR/PadChest. "
        "**NOT a diagnosis** — always have a qualified radiologist interpret "
        "medical images."
    ),
)
async def classify_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray (PNG/JPEG, max 20 MB)."),
) -> ChestXRayResponse:
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                f"Allowed: {sorted(ALLOWED_MIME)}"
            ),
        )
    image_bytes = await file.read()
    return service.process(image_bytes)


@router.get("/info", summary="Module info & status")
async def info() -> dict:
    return {
        "version": "0.1.0",
        "tasks": ["chest_xray"],
        "model_ready": service.is_ready(),
        "max_file_size_mb": 20,
        "supported_mime_types": sorted(ALLOWED_MIME),
    }