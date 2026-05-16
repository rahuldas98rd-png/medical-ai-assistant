"""Schemas for chest X-ray classification."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PathologyPrediction(BaseModel):
    name: str = Field(description="Pathology name (e.g., 'Pneumonia').")
    probability: float = Field(ge=0, le=1)
    confidence_level: Literal["low", "moderate", "high"]
    description: str = ""
    heatmap_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG of the Grad-CAM overlay, if generated.",
    )


class ViewConfidence(BaseModel):
    spread: float = Field(description="Standard deviation of all pathology probabilities.")
    uncertain_count: int = Field(description="How many pathologies fall in [0.4, 0.6].")
    likely_frontal_view: bool
    warning: Optional[str] = None


class ChestXRayResponse(BaseModel):
    predictions: list[PathologyPrediction]
    top_findings: list[PathologyPrediction]
    model_name: str
    image_size_processed: str
    processing_time_ms: int
    input_format: str = Field(description="'image' or 'dicom'.")
    view_confidence: ViewConfidence
    disclaimer: str = (
        "This output is an AI classification of public chest X-ray datasets. "
        "It is NOT a diagnosis. Chest X-ray interpretation requires a qualified "
        "radiologist. AI models inherit demographic and acquisition-related "
        "biases from their training data, and their confidence scores are not "
        "calibrated for clinical decision-making. Always consult a physician "
        "and radiologist for proper diagnosis."
    )