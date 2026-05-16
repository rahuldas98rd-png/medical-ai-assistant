"""
Tests for hypertension, heart disease, and liver disease prediction modules.

Covers:
  - Schema validation (field bounds, enum constraints)
  - Score → risk-level mapping
  - Feature-vector ordering (must match training order)
  - Model-not-loaded raises ModelNotLoadedError
"""

import numpy as np
import pytest
from pydantic import ValidationError

from backend.modules.manual_diagnosis.hypertension_service import (
    FEATURE_NAMES as HTN_FEATURES,
    HIGH_THRESHOLD as HTN_HIGH,
    LOW_THRESHOLD as HTN_LOW,
    HypertensionService,
)
from backend.modules.manual_diagnosis.heart_disease_service import (
    FEATURE_NAMES as HD_FEATURES,
    HIGH_THRESHOLD as HD_HIGH,
    LOW_THRESHOLD as HD_LOW,
    HeartDiseaseService,
)
from backend.modules.manual_diagnosis.liver_disease_service import (
    FEATURE_NAMES as LIVER_FEATURES,
    HIGH_THRESHOLD as LIVER_HIGH,
    LOW_THRESHOLD as LIVER_LOW,
    LiverDiseaseService,
)
from backend.modules.manual_diagnosis.schemas.hypertension import HypertensionRiskRequest
from backend.modules.manual_diagnosis.schemas.heart_disease import HeartDiseaseRiskRequest
from backend.modules.manual_diagnosis.schemas.liver_disease import LiverDiseaseRiskRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_htn_request(**overrides):
    base = dict(
        age=45, systolic_bp=130.0, ldl_cholesterol=3.5,
        adiposity=25.0, family_history=False, type_a_behavior=50,
        obesity_index=26.0, alcohol_units_week=7.0, tobacco_kg_lifetime=0.0,
    )
    base.update(overrides)
    return HypertensionRiskRequest(**base)


def make_hd_request(**overrides):
    base = dict(
        age=54, sex=1, chest_pain_type=2, resting_bp=130.0,
        cholesterol=250.0, fasting_blood_sugar_gt120=0, resting_ecg=1,
        max_heart_rate=160.0, exercise_angina=0, st_depression=1.5,
        st_slope=1, num_major_vessels=0, thalassemia=1,
    )
    base.update(overrides)
    return HeartDiseaseRiskRequest(**base)


def make_liver_request(**overrides):
    base = dict(
        age=45, gender=1, total_bilirubin=0.7, direct_bilirubin=0.1,
        alkaline_phosphotase=187.0, alamine_aminotransferase=16.0,
        aspartate_aminotransferase=18.0, total_proteins=6.8,
        albumin=3.3, albumin_globulin_ratio=0.9,
    )
    base.update(overrides)
    return LiverDiseaseRiskRequest(**base)


# ── Hypertension: schema ──────────────────────────────────────────────────────

class TestHypertensionSchema:
    def test_valid_request(self):
        req = make_htn_request()
        assert req.systolic_bp == 130.0

    def test_age_below_min_rejected(self):
        with pytest.raises(ValidationError):
            make_htn_request(age=0)

    def test_age_above_max_rejected(self):
        with pytest.raises(ValidationError):
            make_htn_request(age=101)

    def test_systolic_bp_too_low_rejected(self):
        with pytest.raises(ValidationError):
            make_htn_request(systolic_bp=79.0)

    def test_systolic_bp_too_high_rejected(self):
        with pytest.raises(ValidationError):
            make_htn_request(systolic_bp=261.0)

    def test_ldl_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            make_htn_request(ldl_cholesterol=0.4)

    def test_negative_tobacco_rejected(self):
        with pytest.raises(ValidationError):
            make_htn_request(tobacco_kg_lifetime=-1.0)

    def test_family_history_bool(self):
        req = make_htn_request(family_history=True)
        assert req.family_history is True


# ── Hypertension: risk mapping ────────────────────────────────────────────────

class TestHypertensionRisk:
    @pytest.mark.parametrize("proba,expected", [
        (0.0,            "low"),
        (HTN_LOW - 0.001, "low"),
        (HTN_LOW,        "moderate"),
        (0.40,           "moderate"),
        (HTN_HIGH - 0.001, "moderate"),
        (HTN_HIGH,       "high"),
        (0.99,           "high"),
    ])
    def test_risk_levels(self, proba, expected):
        svc = HypertensionService()
        assert svc._score_to_risk(proba).label == expected


# ── Hypertension: feature ordering ───────────────────────────────────────────

class TestHypertensionFeatureVector:
    def test_shape_and_order(self):
        svc = HypertensionService()
        req = make_htn_request()
        x = svc._to_feature_vector(req)
        assert x.shape == (9,)
        assert HTN_FEATURES == [
            "systolic_bp", "tobacco_kg_lifetime", "ldl_cholesterol",
            "adiposity", "family_history", "type_a_behavior",
            "obesity_index", "alcohol_units_week", "age",
        ]

    def test_family_history_cast_to_float(self):
        svc = HypertensionService()
        x = svc._to_feature_vector(make_htn_request(family_history=True))
        assert x[4] == 1.0


# ── Hypertension: model-not-loaded ────────────────────────────────────────────

def test_hypertension_raises_when_model_not_loaded():
    from backend.core.exceptions import ModelNotLoadedError
    svc = HypertensionService()
    with pytest.raises(ModelNotLoadedError):
        svc.predict(make_htn_request())


# ── Heart Disease: schema ─────────────────────────────────────────────────────

class TestHeartDiseaseSchema:
    def test_valid_request(self):
        req = make_hd_request()
        assert req.age == 54

    def test_invalid_sex_rejected(self):
        with pytest.raises(ValidationError):
            make_hd_request(sex=2)

    def test_invalid_chest_pain_type_rejected(self):
        with pytest.raises(ValidationError):
            make_hd_request(chest_pain_type=4)

    def test_invalid_resting_ecg_rejected(self):
        with pytest.raises(ValidationError):
            make_hd_request(resting_ecg=3)

    def test_max_heart_rate_too_low(self):
        with pytest.raises(ValidationError):
            make_hd_request(max_heart_rate=59.0)

    def test_cholesterol_too_low(self):
        with pytest.raises(ValidationError):
            make_hd_request(cholesterol=99.0)

    def test_st_depression_too_high(self):
        with pytest.raises(ValidationError):
            make_hd_request(st_depression=10.1)

    def test_invalid_st_slope_rejected(self):
        with pytest.raises(ValidationError):
            make_hd_request(st_slope=3)

    def test_invalid_num_major_vessels(self):
        with pytest.raises(ValidationError):
            make_hd_request(num_major_vessels=4)

    def test_invalid_thalassemia(self):
        with pytest.raises(ValidationError):
            make_hd_request(thalassemia=3)


# ── Heart Disease: risk mapping ───────────────────────────────────────────────

class TestHeartDiseaseRisk:
    @pytest.mark.parametrize("proba,expected", [
        (0.0,            "low"),
        (HD_LOW - 0.001,  "low"),
        (HD_LOW,          "moderate"),
        (0.40,            "moderate"),
        (HD_HIGH - 0.001, "moderate"),
        (HD_HIGH,         "high"),
        (0.99,            "high"),
    ])
    def test_risk_levels(self, proba, expected):
        svc = HeartDiseaseService()
        assert svc._score_to_risk(proba).label == expected


# ── Heart Disease: feature ordering ──────────────────────────────────────────

class TestHeartDiseaseFeatureVector:
    def test_shape_and_order(self):
        svc = HeartDiseaseService()
        x = svc._to_feature_vector(make_hd_request())
        assert x.shape == (13,)
        assert HD_FEATURES == [
            "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
            "fasting_blood_sugar_gt120", "resting_ecg", "max_heart_rate",
            "exercise_angina", "st_depression", "st_slope",
            "num_major_vessels", "thalassemia",
        ]


# ── Heart Disease: model-not-loaded ──────────────────────────────────────────

def test_heart_disease_raises_when_model_not_loaded():
    from backend.core.exceptions import ModelNotLoadedError
    svc = HeartDiseaseService()
    with pytest.raises(ModelNotLoadedError):
        svc.predict(make_hd_request())


# ── Liver Disease: schema ─────────────────────────────────────────────────────

class TestLiverDiseaseSchema:
    def test_valid_request(self):
        req = make_liver_request()
        assert req.age == 45

    def test_invalid_gender_rejected(self):
        with pytest.raises(ValidationError):
            make_liver_request(gender=2)

    def test_total_bilirubin_too_low(self):
        with pytest.raises(ValidationError):
            make_liver_request(total_bilirubin=0.09)

    def test_albumin_too_high(self):
        with pytest.raises(ValidationError):
            make_liver_request(albumin=6.1)

    def test_agr_too_low(self):
        with pytest.raises(ValidationError):
            make_liver_request(albumin_globulin_ratio=0.09)

    def test_alkaline_phosphotase_too_high(self):
        with pytest.raises(ValidationError):
            make_liver_request(alkaline_phosphotase=2501.0)


# ── Liver Disease: risk mapping ───────────────────────────────────────────────

class TestLiverDiseaseRisk:
    @pytest.mark.parametrize("proba,expected", [
        (0.0,               "low"),
        (LIVER_LOW - 0.001, "low"),
        (LIVER_LOW,         "moderate"),
        (0.50,              "moderate"),
        (LIVER_HIGH - 0.001, "moderate"),
        (LIVER_HIGH,        "high"),
        (0.99,              "high"),
    ])
    def test_risk_levels(self, proba, expected):
        svc = LiverDiseaseService()
        assert svc._score_to_risk(proba).label == expected


# ── Liver Disease: feature ordering ──────────────────────────────────────────

class TestLiverDiseaseFeatureVector:
    def test_shape_and_order(self):
        svc = LiverDiseaseService()
        x = svc._to_feature_vector(make_liver_request())
        assert x.shape == (10,)
        assert LIVER_FEATURES == [
            "age", "gender", "total_bilirubin", "direct_bilirubin",
            "alkaline_phosphotase", "alamine_aminotransferase",
            "aspartate_aminotransferase", "total_proteins",
            "albumin", "albumin_globulin_ratio",
        ]


# ── Liver Disease: model-not-loaded ──────────────────────────────────────────

def test_liver_disease_raises_when_model_not_loaded():
    from backend.core.exceptions import ModelNotLoadedError
    svc = LiverDiseaseService()
    with pytest.raises(ModelNotLoadedError):
        svc.predict(make_liver_request())


# ── Recommendations: spot-check clinical triggers ────────────────────────────

class TestHypertensionRecommendations:
    def _get_recs(self, **overrides):
        svc = HypertensionService()
        req = make_htn_request(**overrides)
        risk = svc._score_to_risk(0.6)
        return svc._recommendations(req, risk)

    def test_high_systolic_triggers_hypertension_rec(self):
        recs = self._get_recs(systolic_bp=145.0)
        assert any("140" in r for r in recs)

    def test_elevated_systolic_triggers_monitor_rec(self):
        recs = self._get_recs(systolic_bp=125.0)
        assert any("120" in r for r in recs)

    def test_high_ldl_triggers_cholesterol_rec(self):
        recs = self._get_recs(ldl_cholesterol=4.5)
        assert any("LDL" in r for r in recs)

    def test_smoking_triggers_cessation_rec(self):
        recs = self._get_recs(tobacco_kg_lifetime=5.0)
        assert any("Tobacco" in r or "tobacco" in r.lower() for r in recs)

    def test_non_low_risk_inserts_consult_rec_first(self):
        svc = HypertensionService()
        req = make_htn_request()
        risk = svc._score_to_risk(0.8)
        recs = svc._recommendations(req, risk)
        assert "physician" in recs[0].lower() or "consult" in recs[0].lower()


class TestLiverDiseaseRecommendations:
    def _get_recs(self, **overrides):
        svc = LiverDiseaseService()
        req = make_liver_request(**overrides)
        risk = svc._score_to_risk(0.5)
        return svc._recommendations(req, risk)

    def test_elevated_bilirubin_triggers_rec(self):
        recs = self._get_recs(total_bilirubin=2.0)
        assert any("bilirubin" in r.lower() for r in recs)

    def test_elevated_alt_triggers_rec(self):
        recs = self._get_recs(alamine_aminotransferase=100.0)
        assert any("ALT" in r for r in recs)

    def test_low_albumin_triggers_rec(self):
        recs = self._get_recs(albumin=2.5)
        assert any("albumin" in r.lower() for r in recs)
