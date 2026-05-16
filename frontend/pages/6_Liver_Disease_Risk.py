"""Liver disease risk prediction page (Indian Liver Patient Dataset)."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Liver Disease Risk", page_icon="👨‍🦽", layout="wide")

st.title("🚑 Liver Disease Risk Estimator")
st.caption(
    "Estimate liver disease risk from Liver Function Test (LFT) values. "
    "**This is not a diagnosis.** Consult a gastroenterologist for interpretation of your LFT panel."
)

# Reference range helper displayed in sidebar
with st.sidebar:
    st.markdown("### Normal LFT reference ranges")
    st.markdown(
        "| Marker | Normal range |\n"
        "|---|---|\n"
        "| Total Bilirubin | 0.2–1.2 mg/dL |\n"
        "| Direct Bilirubin | 0.0–0.3 mg/dL |\n"
        "| Alkaline Phosphotase | 44–147 IU/L |\n"
        "| ALT (SGPT) | 7–56 IU/L |\n"
        "| AST (SGOT) | 10–40 IU/L |\n"
        "| Total Proteins | 6.3–8.2 g/dL |\n"
        "| Albumin | 3.5–5.0 g/dL |\n"
        "| A/G Ratio | 1.0–2.5 |"
    )

with st.form("liver_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])

        st.subheader("Bilirubin")
        total_bilirubin = st.number_input(
            "Total Bilirubin (mg/dL)",
            min_value=0.1, max_value=75.0, value=0.7, step=0.1,
            help="Normal: 0.2–1.2 mg/dL. High bilirubin causes jaundice.",
        )
        direct_bilirubin = st.number_input(
            "Direct Bilirubin (mg/dL)",
            min_value=0.0, max_value=20.0, value=0.1, step=0.05,
            help="Normal: 0.0–0.3 mg/dL. High direct bilirubin → conjugated hyperbilirubinaemia.",
        )

        st.subheader("Enzymes")
        alkaline_phosphotase = st.number_input(
            "Alkaline Phosphotase (IU/L)",
            min_value=20.0, max_value=2500.0, value=187.0, step=1.0,
            help="Normal: 44–147 IU/L. Elevated in liver / bone disease.",
        )

    with col2:
        st.subheader("Liver enzymes (transaminases)")
        alamine_aminotransferase = st.number_input(
            "ALT / SGPT (IU/L)",
            min_value=1.0, max_value=2000.0, value=16.0, step=1.0,
            help="Normal: 7–56 IU/L. Most specific marker for hepatocellular damage.",
        )
        aspartate_aminotransferase = st.number_input(
            "AST / SGOT (IU/L)",
            min_value=1.0, max_value=5000.0, value=18.0, step=1.0,
            help="Normal: 10–40 IU/L. Elevated in liver, heart, or muscle disease.",
        )

        st.subheader("Proteins")
        total_proteins = st.number_input(
            "Total Proteins (g/dL)",
            min_value=2.0, max_value=12.0, value=6.8, step=0.1,
            help="Normal: 6.3–8.2 g/dL.",
        )
        albumin = st.number_input(
            "Albumin (g/dL)",
            min_value=0.5, max_value=6.0, value=3.3, step=0.1,
            help="Normal: 3.5–5.0 g/dL. Low albumin → poor liver synthetic function.",
        )
        albumin_globulin_ratio = st.number_input(
            "Albumin / Globulin Ratio (A/G)",
            min_value=0.1, max_value=3.0, value=0.9, step=0.05,
            help="Normal: 1.0–2.5. Ratio <1 may indicate liver or autoimmune disease.",
        )

    submitted = st.form_submit_button("Estimate risk", type="primary", use_container_width=True)

if submitted:
    payload = {
        "age": int(age),
        "gender": 1 if gender == "Male" else 0,
        "total_bilirubin": float(total_bilirubin),
        "direct_bilirubin": float(direct_bilirubin),
        "alkaline_phosphotase": float(alkaline_phosphotase),
        "alamine_aminotransferase": float(alamine_aminotransferase),
        "aspartate_aminotransferase": float(aspartate_aminotransferase),
        "total_proteins": float(total_proteins),
        "albumin": float(albumin),
        "albumin_globulin_ratio": float(albumin_globulin_ratio),
    }

    with st.spinner("Estimating risk..."):
        try:
            resp = requests.post(
                f"{BACKEND}/manual_diagnosis/liver_disease",
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
                    {"range": [0, 35], "color": "#d1fae5"},
                    {"range": [35, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.info(risk["description"])
        if data.get("top_contributors"):
            st.markdown("**Most influential LFT markers:**")
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
