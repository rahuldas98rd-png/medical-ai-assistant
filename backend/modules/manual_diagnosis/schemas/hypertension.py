"""Request / response schemas for hypertension / cardiovascular risk."""

from typing import Literal

from pydantic import BaseModel, Field


class HypertensionRiskRequest(BaseModel):
    """
    Inputs based on the South African Heart Disease study (SAheart dataset).
    Features are clinically established predictors of cardiovascular / hypertension risk.
    """

    age: int = Field(
        ge=1, le=100,
        description="Age in years.",
        examples=[45],
    )
    systolic_bp: float = Field(
        ge=80, le=260,
        description="Current systolic blood pressure (mm Hg). Normal: <120.",
        examples=[130.0],
    )
    ldl_cholesterol: float = Field(
        ge=0.5, le=15.0,
        description="LDL cholesterol (mmol/L). Optimal: <2.6. High: >4.1.",
        examples=[4.5],
    )
    adiposity: float = Field(
        ge=5.0, le=60.0,
        description="Body fat percentage (%). Healthy adult range: 10–30%.",
        examples=[25.0],
    )
    family_history: bool = Field(
        description="First-degree relative with coronary heart disease or hypertension.",
        examples=[False],
    )
    type_a_behavior: int = Field(
        ge=0, le=100,
        description=(
            "Type-A behavior score (coronary-prone personality: hostility, urgency, "
            "competitiveness). Higher = more Type-A. Typical range 13–78."
        ),
        examples=[50],
    )
    obesity_index: float = Field(
        ge=10.0, le=60.0,
        description="Obesity index (similar to BMI but on a different scale). Normal: 22–27.",
        examples=[26.0],
    )
    alcohol_units_week: float = Field(
        ge=0.0, le=200.0,
        description="Alcohol consumption (units per week). 1 unit ≈ 8 g pure alcohol.",
        examples=[7.0],
    )
    tobacco_kg_lifetime: float = Field(
        ge=0.0, le=100.0,
        description="Cumulative lifetime tobacco consumption (kg). Non-smoker = 0.",
        examples=[0.0],
    )


class RiskLevel(BaseModel):
    label: Literal["low", "moderate", "high"]
    score: float = Field(ge=0, le=1, description="Probability of cardiovascular event.")
    description: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float


class HypertensionRiskResponse(BaseModel):
    risk: RiskLevel
    top_contributors: list[FeatureContribution]
    recommendations: list[str] = Field(description="General lifestyle guidance. NOT medical advice.")
    model_version: str
    disclaimer: str = (
        "This is an educational cardiovascular risk estimate based on population-level "
        "statistical patterns. It is NOT a hypertension diagnosis. Blood pressure diagnosis "
        "requires multiple readings over time, interpreted by a qualified physician."
    )
