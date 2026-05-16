"""Brain MRI tumor classification page."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Brain MRI", page_icon="🧠", layout="wide")

st.title("🧠 Brain MRI Tumor Classifier")
st.caption(
    "Classify brain MRI images into 4 categories: glioma, meningioma, pituitary adenoma, or no tumour. "
    "**Educational use only — NOT a diagnosis.** Always consult a radiologist and neurosurgeon."
)

with st.container(border=True):
    st.warning(
        "AI predictions on medical images cannot replace specialist interpretation. "
        "Brain tumour diagnosis requires the full clinical context, multi-sequence MRI, "
        "and evaluation by a qualified radiologist and neurosurgeon.",
        icon="⚠️",
    )

# Check model readiness before accepting uploads
with st.spinner("Checking model status..."):
    try:
        info_resp = requests.get(f"{BACKEND}/brain_mri/info", timeout=5)
        model_ready = info_resp.status_code == 200 and info_resp.json().get("model_ready", False)
    except Exception:
        model_ready = False

if not model_ready:
    st.error(
        "**Brain MRI model is not yet trained.**\n\n"
        "To enable this module:\n"
        "1. Open `ml_training/train_brain_mri.ipynb` in Google Colab\n"
        "2. Run all cells (free T4 GPU, ~30 min training)\n"
        "3. The notebook will push the model to your HuggingFace Hub\n"
        "4. Set `HUGGINGFACE_TOKEN` in your `.env` file\n"
        "5. Restart the backend — the model downloads automatically"
    )
    with st.expander("Dataset: Kaggle Brain MRI Images"):
        st.markdown(
            "- **Dataset**: 'Brain MRI Images for Brain Tumor Detection' by Navoneel Chakrabarty\n"
            "- **Size**: ~3060 MRI images across 4 classes\n"
            "- **Classes**: glioma, meningioma, pituitary adenoma, no_tumor\n"
            "- **Kaggle**: search 'Brain MRI Images for Brain Tumor Detection'"
        )
    st.stop()

uploaded = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a brain MRI image to begin. Use axial, coronal, or sagittal T1/T2 slices.")
    st.stop()

col_img, col_result = st.columns([1, 1])

with col_img:
    st.subheader("Input")
    st.image(uploaded, use_container_width=True)
    run = st.button("Classify", type="primary", use_container_width=True)

if not run:
    st.stop()

with col_result:
    st.subheader("Classification")
    with st.spinner("Running inference (CPU, ~2-5 s)..."):
        try:
            resp = requests.post(
                f"{BACKEND}/brain_mri/classify",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Backend error ({resp.status_code}): {resp.text}")
        st.stop()

    data = resp.json()
    top = data["top_prediction"]

    label_colors = {
        "glioma":     "#ef4444",
        "meningioma": "#f59e0b",
        "notumor":    "#10b981",
        "pituitary":  "#3b82f6",
    }
    color = label_colors.get(top["label"], "#6b7280")
    label_display = top["label"].replace("_", " ").title()

    st.markdown(f"### Top prediction: **{label_display}**")
    st.markdown(f"Confidence: `{top['probability']:.1%}`")
    st.info(top["description"])

# Full probability chart (full width)
st.markdown("---")
st.subheader("All class probabilities")
preds = data["predictions"]

fig = go.Figure(go.Bar(
    x=[p["probability"] for p in preds],
    y=[p["label"].replace("_", " ").title() for p in preds],
    orientation="h",
    marker_color=[label_colors.get(p["label"], "#6b7280") for p in preds],
    text=[f"{p['probability']:.1%}" for p in preds],
    textposition="outside",
))
fig.update_layout(
    height=250,
    margin=dict(l=20, r=60, t=10, b=20),
    xaxis_range=[0, 1.0],
    xaxis_tickformat=".0%",
    yaxis={"categoryorder": "total ascending"},
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model details"):
    st.markdown(f"**Model**: `{data['model_name']}`")
    st.markdown(f"**Image processed at**: `{data['image_size_processed']}`")
    st.caption(f"Inference: `{data['processing_time_ms']} ms`")

st.warning(data["disclaimer"], icon="⚠️")
