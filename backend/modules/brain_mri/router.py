"""Routes for the Brain MRI tumor classification module."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.modules.brain_mri.classifier import MODEL_VERSION, classifier
from backend.modules.brain_mri.schemas.mri import BrainMRIResponse, TumorPrediction

router = APIRouter()

ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post(
    "/classify",
    response_model=BrainMRIResponse,
    summary="Classify a brain MRI image into 4 tumour categories",
)
async def classify_brain_mri(
    file: UploadFile = File(..., description="Brain MRI image (PNG/JPEG), axial/coronal/sagittal."),
) -> BrainMRIResponse:
    if not classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Brain MRI model is not yet available. "
                "Train it via ml_training/train_brain_mri.ipynb (Google Colab), "
                "push to HuggingFace Hub, set HUGGINGFACE_TOKEN in .env, then restart."
            ),
        )
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type: {file.content_type}. Allowed: {sorted(ALLOWED_MIME)}",
        )
    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    predictions_raw, latency_ms = classifier.predict(file_bytes)
    predictions = [TumorPrediction(**p) for p in predictions_raw]
    return BrainMRIResponse(
        predictions=predictions,
        top_prediction=predictions[0],
        model_name=f"ResNet-50 (brain-mri-v{MODEL_VERSION})",
        image_size_processed=f"{224}x{224}",
        processing_time_ms=latency_ms,
    )


@router.get("/info", summary="Module status and model readiness")
async def info() -> dict:
    return {
        "version": MODEL_VERSION,
        "model_ready": classifier.is_ready(),
        "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "architecture": "ResNet-50 (transfer learning, ImageNet init)",
        "training_notebook": "ml_training/train_brain_mri.ipynb",
        "hf_repo": "medimind-ai/brain-mri-resnet50",
    }
