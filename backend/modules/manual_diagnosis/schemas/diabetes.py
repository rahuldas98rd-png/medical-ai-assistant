"""
Request and response schemas for the manual diagnosis module.

Using Pydantic for both validation and auto-generated OpenAPI docs.
Every field carries clinical context (units, normal ranges) so the API
docs double as user-facing documentation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DiabetesRiskRequest(BaseModel):
    """
    Inputs match the Pima Indians Diabetes dataset features. These are the
    8 measurements clinically associated with type 2 diabetes risk.
    """

    pregnancies: int = Field(
        ge=0, le=20,
        description="Number of times pregnant (0 for males / not applicable).",
        examples=[2],
    )
    glucose: float = Field(
        ge=0, le=300,
        description="Plasma glucose concentration after 2hr OGTT (mg/dL). Normal: <140.",
        examples=[120.0],
    )
    blood_pressure: float = Field(
        ge=0, le=200,
        description="Diastolic blood pressure (mm Hg). Normal: <80.",
        examples=[70.0],
    )
    skin_thickness: float = Field(
        ge=0, le=100,
        description="Triceps skin fold thickness (mm). Used as proxy for body fat.",
        examples=[20.0],
    )
    insulin: float = Field(
        ge=0, le=900,
        description="2-Hour serum insulin (mu U/ml). Normal fasting: 16–166.",
        examples=[80.0],
    )
    bmi: float = Field(
        ge=10, le=70,
        description="Body Mass Index (weight kg / height m²). Normal: 18.5–24.9.",
        examples=[28.5],
    )
    diabetes_pedigree: float = Field(
        ge=0, le=3,
        description="Diabetes Pedigree Function — genetic risk score based on family history.",
        examples=[0.45],
    )
    age: int = Field(
        ge=1, le=120,
        description="Age in years.",
        examples=[35],
    )


class RiskLevel(BaseModel):
    label: Literal["low", "moderate", "high"]
    score: float = Field(ge=0, le=1, description="Probability of positive class (diabetes).")
    description: str


class FeatureContribution(BaseModel):
    """Per-feature contribution to this specific prediction (for explainability)."""

    feature: str
    value: float
    importance: float = Field(description="Relative importance of this feature in the model.")


class DiabetesRiskResponse(BaseModel):
    risk: RiskLevel
    top_contributors: list[FeatureContribution] = Field(
        description="Features that most influenced this prediction, descending."
    )
    recommendations: list[str] = Field(
        description="General lifestyle suggestions. NOT medical advice."
    )
    model_version: str
    disclaimer: str = (
        "This prediction is an educational risk estimate, NOT a medical diagnosis. "
        "Diabetes diagnosis requires laboratory tests (HbA1c, FPG, OGTT) interpreted "
        "by a qualified physician. Please consult a healthcare provider."
    )
