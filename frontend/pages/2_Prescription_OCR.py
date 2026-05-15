"""Prescription OCR page."""

import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Prescription OCR", page_icon="📋", layout="wide")

st.title("📋 Prescription Analyzer")
st.caption(
    "Upload a prescription image to extract medicines, dosages, frequencies, "
    "and instructions. **OCR can make mistakes — always verify with the original.**"
)

uploaded = st.file_uploader(
    "Choose a prescription image or PDF",
    type=["jpg", "jpeg", "png", "webp", "bmp", "pdf"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a prescription image to begin.")
    with st.expander("Tips for best results", expanded=True):
        st.markdown(
            "- **Good lighting** — avoid shadows and glare.\n"
            "- **Flat, focused image** — keep the camera parallel to the paper.\n"
            "- **High resolution** — at least 1000 px wide.\n"
            "- **Printed text works best** — v0.1.0 uses Tesseract, which is "
            "optimized for printed text. Handwriting support is planned for v0.2.0."
        )
    st.stop()


col_img, col_result = st.columns([1, 1])

with col_img:
    st.subheader("File")
    if uploaded.type == "application/pdf":
        st.info(f"📄 PDF file: **{uploaded.name}** ({uploaded.size / 1024:.0f} KB)")
        st.caption(
            "PDF preview isn't shown inline. Click Extract to process — digital "
            "PDFs are read directly (fast); scanned PDFs go through OCR (slower)."
        )
    else:
        st.image(uploaded, use_container_width=True)
    run = st.button("Extract", type="primary", use_container_width=True)

if not run:
    st.stop()

with col_result:
    st.subheader("Extracted information")
    with st.spinner("Running OCR..."):
        try:
            resp = requests.post(
                f"{BACKEND}/prescription_ocr/extract",
                files={
                    "file": (uploaded.name, uploaded.getvalue(), uploaded.type),
                },
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Backend error ({resp.status_code}): {resp.text}")
        st.stop()

    data = resp.json()
    ext = data["extraction"]

    # ---- Header info ----
    header_present = any(
        [ext.get("patient_name"), ext.get("patient_age"),
         ext.get("doctor_name"), ext.get("prescription_date")]
    )
    if header_present:
        st.markdown("**Header**")
        if ext.get("patient_name"):
            st.markdown(f"- Patient: `{ext['patient_name']}`")
        if ext.get("patient_age"):
            st.markdown(f"- Age: `{ext['patient_age']}`")
        if ext.get("doctor_name"):
            st.markdown(f"- Doctor: `Dr. {ext['doctor_name']}`")
        if ext.get("prescription_date"):
            st.markdown(f"- Date: `{ext['prescription_date']}`")

    # ---- Medicines table ----
    if ext["medicines"]:
        st.markdown("**Medicines**")
        st.dataframe(
            ext["medicines"],
            column_config={
                "name": st.column_config.TextColumn("Medicine", width="medium"),
                "dosage": "Dosage",
                "frequency": "Frequency",
                "duration": "Duration",
                "raw_line": st.column_config.TextColumn(
                    "Source OCR line", width="large", help="Verify against original"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    # ---- Instructions ----
    if ext["general_instructions"]:
        st.markdown("**Instructions**")
        for line in ext["general_instructions"]:
            st.markdown(f"- {line}")

    # ---- Warnings ----
    for w in ext.get("confidence_warnings", []):
        st.warning(w)

# ---- Raw OCR + meta (full width) ----
with st.expander("Raw OCR output (verify against original here)"):
    st.text(data["raw_text"])
    st.caption(
        f"Engine: `{data['ocr_engine']}` · "
        f"Processing time: `{data['processing_time_ms']} ms`"
    )

st.warning(data["disclaimer"], icon="⚠️")