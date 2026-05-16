"""Chest X-ray classification page (v0.2.0): heatmaps, DICOM, view warning."""

import base64
import io
import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Chest X-Ray", page_icon="🫁", layout="wide")

st.title("🫁 Chest X-Ray Classifier")
st.caption(
    "Multi-label pathology classification with Grad-CAM heatmaps. "
    "Supports image (PNG/JPEG) and DICOM input. **Educational use only — NOT a diagnosis.**"
)

with st.container(border=True):
    st.warning(
        "AI predictions on medical images cannot replace radiologist "
        "interpretation. Pretrained models have demographic and acquisition "
        "biases. Always consult a qualified radiologist.",
        icon="⚠️",
    )

uploaded = st.file_uploader(
    "Upload a chest X-ray (image or DICOM)",
    type=["jpg", "jpeg", "png", "webp", "bmp", "dcm", "dicom"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a chest X-ray to begin.")
    with st.expander("Where to find test X-rays"):
        st.markdown(
            "- **Wikimedia Commons**: search 'Chest radiograph' or 'Pneumonia x-ray' "
            "for public-domain examples (use frontal views — lateral views give "
            "poor results, which the view-confidence warning will flag).\n"
            "- **Kaggle 'Chest X-Ray Images Pneumonia'**: ~5800 labeled samples.\n"
            "- **NIH ChestX-ray14** on Kaggle for more variety."
        )
    st.stop()


col_img, col_result = st.columns([1, 1])

with col_img:
    st.subheader("Input")
    if uploaded.name.lower().endswith((".dcm", ".dicom")):
        st.info(f"📋 DICOM file: **{uploaded.name}** ({uploaded.size / 1024:.0f} KB)")
        st.caption("Preview not shown inline for DICOM — click Analyze to process.")
    else:
        st.image(uploaded, use_container_width=True)
    run = st.button("Analyze", type="primary", use_container_width=True)

if not run:
    st.stop()

with col_result:
    st.subheader("Pathology probabilities")
    with st.spinner("Running inference + heatmaps (CPU, ~5-12 s)..."):
        try:
            resp = requests.post(
                f"{BACKEND}/medical_imaging/chest_xray",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                timeout=120,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Backend error ({resp.status_code}): {resp.text}")
        st.stop()

    data = resp.json()

    # ---- View confidence warning ----
    view = data["view_confidence"]
    if view.get("warning"):
        st.error(f"⚠️ {view['warning']}")
        st.caption(
            f"Spread: {view['spread']:.3f} · "
            f"Pathologies in [0.4, 0.6]: {view['uncertain_count']}/18"
        )
    else:
        st.success(
            f"View confidence: looks like a frontal X-ray "
            f"(spread {view['spread']:.3f})"
        )

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
        st.success("No pathologies above the 50% threshold.")

# ---- Heatmaps section (full width) ----
heatmap_findings = [f for f in data.get("top_findings", []) if f.get("heatmap_base64")]
if heatmap_findings:
    st.markdown("---")
    st.subheader("Grad-CAM — where the model focused")
    st.caption(
        "Red regions had the strongest influence on each prediction. "
        "Useful for sanity-checking whether the model is looking at anatomically "
        "plausible locations."
    )
    cols = st.columns(min(len(heatmap_findings), 3))
    for i, f in enumerate(heatmap_findings):
        with cols[i % 3]:
            img_bytes = base64.b64decode(f["heatmap_base64"])
            st.image(
                io.BytesIO(img_bytes),
                caption=f"{f['name'].replace('_', ' ')} · {f['probability']:.1%}",
                use_container_width=True,
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
    st.markdown(f"**Input format**: `{data['input_format']}`")
    st.markdown(f"**Image processed at**: `{data['image_size_processed']}`")
    st.caption(f"Inference + heatmaps: `{data['processing_time_ms']} ms`")

st.warning(data["disclaimer"], icon="⚠️")