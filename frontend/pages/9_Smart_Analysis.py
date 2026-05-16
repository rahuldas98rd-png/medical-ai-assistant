"""Smart multi-module health analysis page — Phase 5 orchestrator."""

import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY", "")

# Build a headers dict used for every backend call on this page
_HEADERS: dict = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="Smart Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Smart Health Analysis")
st.caption(
    "Describe your symptoms and provide any available measurements. "
    "The system automatically routes to relevant AI modules and returns a unified screening report."
)

with st.container(border=True):
    st.warning(
        "This tool performs **educational screening only**. It does not diagnose any condition. "
        "Always consult a qualified healthcare professional for medical evaluation.",
        icon="⚠️",
    )

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("analysis_form"):
    col_left, col_right = st.columns([3, 2])

    with col_left:
        symptoms = st.text_area(
            "Describe your symptoms *",
            placeholder=(
                "e.g. I have been experiencing excessive thirst and frequent urination "
                "for the past few weeks. My blood pressure reading was 145/95 last week "
                "and I have occasional headaches."
            ),
            height=160,
        )

        st.markdown("**Optional measurements** — provide what you have:")
        m1, m2, m3 = st.columns(3)
        with m1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=None, placeholder="e.g. 45")
            glucose = st.number_input("Blood glucose (mg/dL)", min_value=30.0, max_value=600.0, value=None, placeholder="e.g. 126")
        with m2:
            gender = st.selectbox("Gender", ["— not specified —", "Male", "Female"])
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60.0, max_value=260.0, value=None, placeholder="e.g. 140")
        with m3:
            bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=None, placeholder="e.g. 27.5")

    with col_right:
        st.markdown("**Optional image upload**")
        image_file = st.file_uploader(
            "Upload a medical image",
            type=["jpg", "jpeg", "png", "bmp", "webp", "dcm"],
            help="Chest X-ray, Brain MRI, or Prescription image",
        )
        image_type = None
        if image_file:
            st.image(image_file, use_container_width=True)
            image_type = st.selectbox(
                "Image type *",
                ["chest_xray", "brain_mri", "prescription"],
                format_func=lambda x: {
                    "chest_xray": "Chest X-Ray",
                    "brain_mri": "Brain MRI",
                    "prescription": "Prescription",
                }[x],
            )

    submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)

if not submitted:
    st.stop()

if not symptoms or len(symptoms.strip()) < 10:
    st.error("Please describe your symptoms (at least 10 characters).")
    st.stop()

# ── Build and send request ────────────────────────────────────────────────────
form_data: dict = {"symptoms": symptoms.strip()}
if age:
    form_data["age"] = str(int(age))
if gender != "— not specified —":
    form_data["gender"] = gender.lower()
if bmi:
    form_data["bmi"] = str(bmi)
if glucose:
    form_data["glucose"] = str(glucose)
if systolic_bp:
    form_data["systolic_bp"] = str(systolic_bp)
if image_type:
    form_data["image_type"] = image_type

files = {}
if image_file:
    files["image"] = (image_file.name, image_file.getvalue(), image_file.type or "image/jpeg")

with st.spinner("Running multi-module analysis…"):
    try:
        resp = requests.post(
            f"{BACKEND}/orchestrator/analyze",
            data=form_data,
            files=files if files else None,
            headers=_HEADERS,
            timeout=120,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Backend unreachable: {e}")
        st.stop()

if resp.status_code == 401:
    st.error("Authentication required. Set the API_KEY environment variable.")
    st.stop()
elif resp.status_code == 429:
    st.error("Rate limit reached (15 requests/minute). Please wait a moment and try again.")
    st.stop()
elif resp.status_code != 200:
    st.error(f"Backend error ({resp.status_code}): {resp.text}")
    st.stop()

data = resp.json()

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Analysis Report")
st.info(data["overall_summary"])

col_a, col_b = st.columns(2)

# ── Condition assessments ─────────────────────────────────────────────────────
with col_a:
    st.markdown("#### Condition Screening")
    assessments = data.get("condition_assessments", [])
    if not assessments:
        st.write("No conditions triggered from the symptom description.")
    for ca in assessments:
        status = ca["status"]
        display = ca["display_name"]
        risk_obj = ca.get("risk") or {}
        icon = {"success": "🔴" if risk_obj.get("label") == "high"
                else "🟡" if risk_obj.get("label") == "moderate"
                else "🟢",
                "flagged": "🔵",
                "error": "⚠️"}.get(status, "⚪")

        with st.expander(f"{icon} {display}  —  {status.upper()}", expanded=(status == "success")):
            kws = ca.get("matched_keywords", [])
            if kws:
                st.caption(f"Triggered by: {', '.join(kws)}")

            if status == "success" and risk_obj:
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    color = {"low": "green", "moderate": "orange", "high": "red"}.get(risk_obj["label"], "gray")
                    st.markdown(f"**Risk:** :{color}[{risk_obj['label'].upper()}]")
                with col_r2:
                    st.metric("Score", f"{risk_obj['score']:.0%}")
                st.caption(risk_obj["description"])

                if ca.get("top_contributors"):
                    st.markdown("**Top contributing factors:**")
                    for c in ca["top_contributors"][:3]:
                        st.write(f"- {c['feature']}: `{c['value']}` (importance: {c['importance']:.3f})")

                if ca.get("recommendations"):
                    st.markdown("**Recommendations:**")
                    for r in ca["recommendations"][:3]:
                        st.write(f"- {r}")

            elif status == "flagged":
                st.write(
                    f"Symptoms suggest **{display}** may be relevant. "
                    f"Visit the dedicated page for a complete assessment with full clinical inputs."
                )
                st.page_link(
                    f"pages/{ca['detail_page']}.py",
                    label=f"Go to {display} →",
                )

            elif status == "error":
                st.warning("Assessment failed — use the dedicated module page directly.")

# ── Image assessment ──────────────────────────────────────────────────────────
with col_b:
    img_data = data.get("image_assessment")
    if img_data:
        itype_display = img_data["image_type"].replace("_", " ").title()
        st.markdown(f"#### {itype_display} Analysis")

        if img_data["status"] == "success":
            tf = img_data.get("top_finding")
            if tf:
                label_display = tf["label"].replace("_", " ").title()
                confidence = tf["probability"]
                badge_color = "red" if confidence > 0.6 else "orange" if confidence > 0.3 else "green"
                st.markdown(f"**Top finding:** :{badge_color}[{label_display}]")
                st.metric("Confidence", f"{confidence:.1%}")
                if tf.get("description"):
                    st.caption(tf["description"])

            if img_data["image_type"] == "prescription":
                medicines = img_data.get("extra", {}).get("medicines", [])
                if medicines:
                    st.markdown("**Extracted medicines:**")
                    for m in medicines:
                        dosage = m.get("dosage", "")
                        freq = m.get("frequency", "")
                        detail = f" — {dosage}" if dosage else ""
                        detail += f", {freq}" if freq else ""
                        st.write(f"- **{m['name']}**{detail}")
                raw = img_data.get("extra", {}).get("raw_text_preview", "")
                if raw:
                    with st.expander("Raw OCR text"):
                        st.code(raw)
            else:
                findings = img_data.get("all_findings", [])
                if len(findings) > 1:
                    with st.expander("All findings"):
                        for f in findings:
                            bar_val = f["probability"]
                            st.write(f"**{f['label'].replace('_',' ').title()}**: {bar_val:.1%}")
                            st.progress(min(bar_val, 1.0))

        elif img_data["status"] == "error":
            st.error(f"Image analysis failed: {img_data.get('extra', {}).get('error', 'Unknown error')}")
    else:
        st.markdown("#### Image Analysis")
        st.write("No image provided. Upload a chest X-ray, brain MRI, or prescription for image-based analysis.")

# ── Key recommendations ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Key Recommendations")
recs = data.get("key_recommendations", [])
for r in recs:
    st.write(f"- {r}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
inputs_used = data.get("inputs_used", {})
if inputs_used:
    st.caption(f"Inputs used: {', '.join(f'{k}={v}' for k, v in inputs_used.items())}")
st.caption(f"Processing time: {data.get('processing_time_ms', 0)} ms")
st.warning(data.get("disclaimer", ""), icon="⚠️")

# ── Consultation history ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Previous Consultations", expanded=False):
    try:
        hist_resp = requests.get(
            f"{BACKEND}/orchestrator/history",
            headers=_HEADERS,
            timeout=10,
        )
        if hist_resp.status_code == 200:
            hist = hist_resp.json()
            consultations = hist.get("consultations", [])
            if not consultations:
                st.write("No previous consultations found for your session.")
            else:
                st.caption(f"Showing last {len(consultations)} consultation(s)")
                for item in consultations:
                    ts = item["created_at"][:19].replace("T", " ")
                    with st.expander(f"🕐 {ts} — {item['symptoms_preview'][:80]}…"):
                        st.write(item.get("overall_summary", ""))
        else:
            st.write("Could not load history.")
    except Exception:
        st.write("History unavailable.")
