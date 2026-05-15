"""Schemas for chest X-ray classification."""

from typing import Literal

from pydantic import BaseModel, Field


class PathologyPrediction(BaseModel):
    name: str = Field(description="Pathology name (e.g., 'Pneumonia').")
    probability: float = Field(ge=0, le=1)
    confidence_level: Literal["low", "moderate", "high"]
    description: str = Field(default="", description="Plain-language explanation.")


class ChestXRayResponse(BaseModel):
    predictions: list[PathologyPrediction] = Field(
        description="All pathology probabilities, sorted descending."
    )
    top_findings: list[PathologyPrediction] = Field(
        description="Predictions above the top-finding threshold (default 0.5)."
    )
    model_name: str
    image_size_processed: str
    processing_time_ms: int
    disclaimer: str = (
        "This output is an AI classification of public chest X-ray datasets. "
        "It is NOT a diagnosis. Chest X-ray interpretation requires a qualified "
        "radiologist. AI models inherit demographic and acquisition-related "
        "biases from their training data, and their confidence scores are not "
        "calibrated for clinical decision-making. Always consult a physician "
        "and radiologist for proper diagnosis."
    )