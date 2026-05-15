"""Chest X-ray classification page."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Chest X-Ray", page_icon="🫁", layout="wide")

st.title("🫁 Chest X-Ray Classifier")
st.caption(
    "Multi-label pathology classification for chest X-rays. "
    "**Educational use only — NOT a diagnosis.**"
)

with st.container(border=True):
    st.warning(
        "The model can flag possible signs of 18 pathologies but cannot replace "
        "clinical interpretation. False positives and negatives are common. "
        "Pretrained models inherit biases from training data (demographics, "
        "acquisition equipment). Always consult a qualified radiologist.",
        icon="⚠️",
    )

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a chest X-ray image to begin.")
    with st.expander("Where to find test X-rays"):
        st.markdown(
            "- **Kaggle**: search 'Chest X-Ray Images Pneumonia' — ~5800 labeled samples.\n"
            "- **NIH ChestX-ray14**: also on Kaggle.\n"
            "- **Wikimedia Commons**: search 'chest radiograph' for public-domain examples.\n\n"
            "Best results with **frontal (PA / AP) views** at reasonable resolution "
            "(≥512px). Lateral views weren't well-represented in training data."
        )
    st.stop()


col_img, col_result = st.columns([1, 1])

with col_img:
    st.subheader("X-ray image")
    st.image(uploaded, use_container_width=True)
    run = st.button("Analyze", type="primary", use_container_width=True)

if not run:
    st.stop()

with col_result:
    st.subheader("Pathology probabilities")
    with st.spinner("Running inference (CPU, ~3-5 s)..."):
        try:
            resp = requests.post(
                f"{BACKEND}/medical_imaging/chest_xray",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Backend error ({resp.status_code}): {resp.text}")
        st.stop()

    data = resp.json()

    # ---- Top findings ----
    top = data.get("top_findings", [])
    if top:
        st.markdown("**Top findings** (probability ≥ 50%):")
        for f in top:
            emoji = {"high": "🔴", "moderate": "🟡", "low": "🟢"}[f["confidence_level"]]
            st.markdown(
                f"{emoji} **{f['name'].replace('_', ' ')}** — `{f['probability']:.1%}`"
            )
            if f.get("description"):
                st.caption(f["description"])
    else:
        st.success(
            "No pathologies detected above the 50% probability threshold. "
            "Always verify with a qualified radiologist."
        )

# ---- Full probability chart ----
st.markdown("---")
st.subheader("All pathology scores")

preds = data["predictions"]
fig = go.Figure(
    go.Bar(
        x=[p["probability"] for p in preds],
        y=[p["name"].replace("_", " ") for p in preds],
        orientation="h",
        marker_color=[
            "#ef4444" if p["probability"] >= 0.7
            else "#f59e0b" if p["probability"] >= 0.3
            else "#94a3b8"
            for p in preds
        ],
        text=[f"{p['probability']:.1%}" for p in preds],
        textposition="outside",
    )
)
fig.update_layout(
    height=520,
    margin=dict(l=20, r=40, t=20, b=20),
    xaxis_range=[0, 1.0],
    xaxis_tickformat=".0%",
    yaxis={"categoryorder": "total ascending"},
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model details"):
    st.markdown(f"**Model**: `{data['model_name']}`")
    st.markdown(f"**Image preprocessed to**: `{data['image_size_processed']}`")
    st.caption(f"Inference time: `{data['processing_time_ms']} ms`")

st.warning(data["disclaimer"], icon="⚠️")