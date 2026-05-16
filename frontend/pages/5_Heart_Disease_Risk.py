"""Heart disease risk prediction page (Cleveland dataset, XGBoost)."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Risk Estimator")
st.caption(
    "Estimate coronary artery disease risk using clinical measurements from the "
    "Cleveland Heart Disease dataset. **This is not a diagnosis.** Consult a cardiologist for proper evaluation."
)

CHEST_PAIN_OPTIONS = {
    "Typical angina (chest pain with exertion, relieved by rest)": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3,
}
ECG_OPTIONS = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
SLOPE_OPTIONS = {
    "Upsloping (normal)": 0,
    "Flat": 1,
    "Downsloping (worse prognosis)": 2,
}
THAL_OPTIONS = {
    "Normal": 0,
    "Fixed defect": 1,
    "Reversible defect": 2,
}

with st.form("heart_disease_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & vitals")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=54, step=1)
        sex = st.selectbox("Biological sex", options=["Male", "Female"])
        resting_bp = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=60.0, max_value=250.0, value=130.0, step=1.0,
            help="Blood pressure recorded on hospital admission.",
        )
        cholesterol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=100.0, max_value=600.0, value=250.0, step=1.0,
            help="Desirable: <200. Borderline: 200–239. High: ≥240.",
        )
        max_heart_rate = st.number_input(
            "Max heart rate achieved (bpm)",
            min_value=60.0, max_value=220.0, value=160.0, step=1.0,
            help="Maximum heart rate during exercise stress test.",
        )

    with col2:
        st.subheader("Cardiac tests & symptoms")
        chest_pain_type = st.selectbox("Chest pain type", options=list(CHEST_PAIN_OPTIONS.keys()))
        fasting_bs = st.checkbox(
            "Fasting blood sugar > 120 mg/dL",
            help="Indicator of pre-diabetes / diabetes risk.",
        )
        resting_ecg = st.selectbox("Resting ECG result", options=list(ECG_OPTIONS.keys()))
        exercise_angina = st.checkbox(
            "Exercise-induced angina",
            help="Chest pain or discomfort that occurs during physical activity.",
        )
        st_depression = st.number_input(
            "ST depression (mm)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="ST depression induced by exercise relative to rest. Normal: 0.",
        )
        st_slope = st.selectbox("ST segment slope (peak exercise)", options=list(SLOPE_OPTIONS.keys()))
        num_vessels = st.selectbox(
            "Major vessels colored by fluoroscopy",
            options=[0, 1, 2, 3],
            help="Number of major coronary vessels with significant stenosis (0 = none).",
        )
        thalassemia = st.selectbox("Thalassemia", options=list(THAL_OPTIONS.keys()))

    submitted = st.form_submit_button("Estimate risk", type="primary", use_container_width=True)

if submitted:
    payload = {
        "age": int(age),
        "sex": 1 if sex == "Male" else 0,
        "chest_pain_type": CHEST_PAIN_OPTIONS[chest_pain_type],
        "resting_bp": float(resting_bp),
        "cholesterol": float(cholesterol),
        "fasting_blood_sugar_gt120": 1 if fasting_bs else 0,
        "resting_ecg": ECG_OPTIONS[resting_ecg],
        "max_heart_rate": float(max_heart_rate),
        "exercise_angina": 1 if exercise_angina else 0,
        "st_depression": float(st_depression),
        "st_slope": SLOPE_OPTIONS[st_slope],
        "num_major_vessels": int(num_vessels),
        "thalassemia": THAL_OPTIONS[thalassemia],
    }

    with st.spinner("Estimating risk..."):
        try:
            resp = requests.post(
                f"{BACKEND}/manual_diagnosis/heart_disease",
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
