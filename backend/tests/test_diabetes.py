"""
Tests for the diabetes prediction module.

These tests don't require a trained model to be present — they cover:
  - schema validation (Pydantic rules catch bad input)
  - score → risk-level mapping
  - feature-vector ordering matches training order
  - module registration via the auto-discovery pipeline

Run from project root:
    pytest backend/tests -v
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.modules.manual_diagnosis.diabetes_service import (
    FEATURE_NAMES,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    DiabetesService,
)
from backend.modules.manual_diagnosis.schemas.diabetes import DiabetesRiskRequest


# -------------------------------------------------------------------------
# Schema validation
# -------------------------------------------------------------------------
def make_valid_request(**overrides):
    base = dict(
        pregnancies=2, glucose=120.0, blood_pressure=70.0, skin_thickness=20.0,
        insulin=80.0, bmi=28.5, diabetes_pedigree=0.45, age=35,
    )
    base.update(overrides)
    return DiabetesRiskRequest(**base)


def test_valid_request_passes():
    req = make_valid_request()
    assert req.glucose == 120.0


def test_negative_glucose_rejected():
    with pytest.raises(ValidationError):
        make_valid_request(glucose=-10)


def test_extreme_age_rejected():
    with pytest.raises(ValidationError):
        make_valid_request(age=200)


def test_pedigree_upper_bound():
    with pytest.raises(ValidationError):
        make_valid_request(diabetes_pedigree=5.0)


# -------------------------------------------------------------------------
# Risk-level mapping
# -------------------------------------------------------------------------
@pytest.mark.parametrize("proba,expected", [
    (0.05, "low"),
    (LOW_THRESHOLD - 0.001, "low"),
    (LOW_THRESHOLD, "moderate"),
    (0.5, "moderate"),
    (HIGH_THRESHOLD - 0.001, "moderate"),
    (HIGH_THRESHOLD, "high"),
    (0.95, "high"),
])
def test_score_to_risk_label(proba, expected):
    svc = DiabetesService()
    assert svc._score_to_risk(proba).label == expected


# -------------------------------------------------------------------------
# Feature ordering
# -------------------------------------------------------------------------
def test_feature_vector_order_matches_training_order():
    """
    Critical: the order of features fed to predict() must EXACTLY match
    the order they were trained on. Drift here would silently produce
    garbage predictions.
    """
    svc = DiabetesService()
    req = make_valid_request()
    x = svc._to_feature_vector(req)
    assert x.shape == (8,)
    assert FEATURE_NAMES == [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age",
    ]


# -------------------------------------------------------------------------
# Module auto-discovery
# -------------------------------------------------------------------------
def test_module_is_discoverable():
    """The registry should pick up the ManualDiagnosisModule automatically."""
    from backend.core.registry import ModuleRegistry

    reg = ModuleRegistry()
    reg.discover()
    names = [m.name for m in reg]
    assert "manual_diagnosis" in names


# -------------------------------------------------------------------------
# Integration: missing model fails gracefully
# -------------------------------------------------------------------------
def test_predict_raises_when_model_not_loaded():
    from backend.core.exceptions import ModelNotLoadedError

    svc = DiabetesService()
    # Don't call svc.load() — model is None
    with pytest.raises(ModelNotLoadedError):
        svc.predict(make_valid_request())
