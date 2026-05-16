"""Schemas for Brain MRI tumor classification."""

from typing import Optional
from pydantic import BaseModel, Field


class TumorPrediction(BaseModel):
    label: str = Field(description="Tumor class: glioma, meningioma, pituitary, or no_tumor.")
    probability: float = Field(ge=0, le=1)
    description: str = ""


class BrainMRIResponse(BaseModel):
    predictions: list[TumorPrediction] = Field(
        description="All four classes with probabilities, sorted descending."
    )
    top_prediction: TumorPrediction
    model_name: str
    image_size_processed: str
    processing_time_ms: int
    disclaimer: str = (
        "This is an AI classification of brain MRI images for educational purposes only. "
        "It is NOT a medical diagnosis. Brain tumor diagnosis requires evaluation by a "
        "qualified radiologist and neurosurgeon using the full clinical context."
    )
