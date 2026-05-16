"""Hypertension / cardiovascular risk prediction page."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Hypertension Risk", page_icon="🫀", layout="wide")

st.title("🫀 Hypertension & Cardiovascular Risk Estimator")
st.caption(
    "Estimate cardiovascular / hypertension risk from clinical measurements. "
    "**This is not a diagnosis.** Always consult a physician for proper blood pressure evaluation."
)

with st.form("hypertension_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Blood pressure & lipids")
        systolic_bp = st.number_input(
            "Systolic blood pressure (mm Hg)",
            min_value=80.0, max_value=260.0, value=130.0, step=1.0,
            help="Normal: <120. Elevated: 120–129. High: ≥130.",
        )
        ldl_cholesterol = st.number_input(
            "LDL cholesterol (mmol/L)",
            min_value=0.5, max_value=15.0, value=3.5, step=0.1,
            help="Optimal: <2.6. Near-optimal: 2.6–3.3. High: >4.1.",
        )
        adiposity = st.number_input(
            "Body fat percentage (%)",
            min_value=5.0, max_value=60.0, value=25.0, step=0.5,
            help="Healthy adult range: men 10–20%, women 18–28%.",
        )
        obesity_index = st.number_input(
            "Obesity index (BMI-scale)",
            min_value=10.0, max_value=60.0, value=26.0, step=0.1,
            help="Normal BMI range: 18.5–24.9. Overweight: 25–29.9. Obese: ≥30.",
        )

    with col2:
        st.subheader("Lifestyle & history")
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=45, step=1)
        family_history = st.checkbox(
            "Family history of heart disease or hypertension",
            value=False,
            help="First-degree relative (parent, sibling) with coronary heart disease or hypertension.",
        )
        tobacco_kg_lifetime = st.number_input(
            "Lifetime tobacco consumption (kg)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.5,
            help="Non-smoker = 0. 1 pack/day for 1 year ≈ 7.3 kg.",
        )
        alcohol_units_week = st.number_input(
            "Alcohol consumption (units/week)",
            min_value=0.0, max_value=200.0, value=5.0, step=1.0,
            help="1 unit ≈ 125 ml wine, 250 ml beer, 25 ml spirits. Low-risk: ≤14 units/week.",
        )
        type_a_behavior = st.slider(
            "Type-A behavior score (0 = relaxed, 100 = highly driven/hostile)",
            min_value=0, max_value=100, value=50,
            help="Higher scores indicate competitive, time-pressured, hostile personality traits "
                 "linked to cardiovascular risk.",
        )

    submitted = st.form_submit_button("Estimate risk", type="primary", use_container_width=True)

if submitted:
    payload = {
        "age": int(age),
        "systolic_bp": float(systolic_bp),
        "ldl_cholesterol": float(ldl_cholesterol),
        "adiposity": float(adiposity),
        "family_history": bool(family_history),
        "type_a_behavior": int(type_a_behavior),
        "obesity_index": float(obesity_index),
        "alcohol_units_week": float(alcohol_units_week),
        "tobacco_kg_lifetime": float(tobacco_kg_lifetime),
    }

    with st.spinner("Estimating risk..."):
        try:
            resp = requests.post(
                f"{BACKEND}/manual_diagnosis/hypertension",
                json=payload,
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Backend error ({resp.status_code}): {resp.text}")
        st.stop()

    data = resp.json()
    risk = data["risk"]
    color = {"low": "#10b981", "moderate": "#f59e0b", "high": "#ef4444"}[risk["label"]]

    st.markdown("### Result")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk["score"] * 100,
            number={"suffix": "%"},
            title={"text": f"Risk: <b>{risk['label'].upper()}</b>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": "#d1fae5"},
                    {"range": [25, 55], "color": "#fef3c7"},
                    {"range": [55, 100], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.info(risk["description"])
        if data.get("top_contributors"):
            st.markdown("**Most influential factors:**")
            for c in data["top_contributors"]:
                st.markdown(
                    f"- **{c['feature'].replace('_', ' ').title()}** — "
                    f"your value: `{c['value']}`, model importance: `{c['importance']:.2%}`"
                )

    st.markdown("### Recommendations")
    for rec in data["recommendations"]:
        st.markdown(f"• {rec}")

    with st.expander("Model & disclaimer"):
        st.caption(f"Model version: `{data['model_version']}`")
        st.warning(data["disclaimer"])
