"""Diabetes risk prediction page."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Diabetes Risk", page_icon="🩸", layout="wide")

st.title("🩸 Diabetes Risk Estimator")
st.caption(
    "Estimate type 2 diabetes risk from clinical measurements. "
    "**This is not a diagnosis.** Always consult a physician for proper testing."
)

# ---- Input form ----
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clinical measurements")
        glucose = st.number_input(
            "Glucose (mg/dL, plasma, 2hr post OGTT)",
            min_value=0.0, max_value=300.0, value=120.0, step=1.0,
            help="Normal post-OGTT: <140 mg/dL. Pre-diabetic: 140-199. Diabetic: ≥200.",
        )
        blood_pressure = st.number_input(
            "Blood pressure (diastolic, mm Hg)",
            min_value=0.0, max_value=200.0, value=70.0, step=1.0,
            help="Normal diastolic: <80 mm Hg",
        )
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0, max_value=70.0, value=28.5, step=0.1,
            help="Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: ≥30",
        )
        insulin = st.number_input(
            "2-Hour serum insulin (mu U/ml)",
            min_value=0.0, max_value=900.0, value=80.0, step=1.0,
        )

    with col2:
        st.subheader("Personal & history")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=35, step=1)
        pregnancies = st.number_input(
            "Number of pregnancies",
            min_value=0, max_value=20, value=0, step=1,
            help="Enter 0 if not applicable.",
        )
        skin_thickness = st.number_input(
            "Triceps skin fold (mm)",
            min_value=0.0, max_value=100.0, value=20.0, step=1.0,
            help="Used as a proxy for body fat.",
        )
        diabetes_pedigree = st.number_input(
            "Diabetes pedigree score",
            min_value=0.0, max_value=3.0, value=0.45, step=0.01,
            help="Family history score: higher = stronger genetic predisposition. "
                 "Typical range 0.08-2.42.",
        )

    submitted = st.form_submit_button("Estimate risk", type="primary", use_container_width=True)


# ---- Prediction ----
if submitted:
    payload = {
        "pregnancies": int(pregnancies),
        "glucose": float(glucose),
        "blood_pressure": float(blood_pressure),
        "skin_thickness": float(skin_thickness),
        "insulin": float(insulin),
        "bmi": float(bmi),
        "diabetes_pedigree": float(diabetes_pedigree),
        "age": int(age),
    }

    with st.spinner("Estimating risk..."):
        try:
            resp = requests.post(
                f"{BACKEND}/manual_diagnosis/diabetes",
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

    # ---- Result display ----
    st.markdown("### Result")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        # Gauge chart for risk
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk["score"] * 100,
            number={"suffix": "%"},
            title={"text": f"Risk: <b>{risk['label'].upper()}</b>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.info(risk["description"])

        if data.get("top_contributors"):
            st.markdown("**Most influential factors in the model:**")
            for c in data["top_contributors"]:
                st.markdown(
                    f"- **{c['feature'].replace('_', ' ').title()}** — "
                    f"your value: `{c['value']}`, model importance: "
                    f"`{c['importance']:.2%}`"
                )

    st.markdown("### Recommendations")
    for rec in data["recommendations"]:
        st.markdown(f"• {rec}")

    with st.expander("Model & disclaimer"):
        st.caption(f"Model version: `{data['model_version']}`")
        st.warning(data["disclaimer"])
