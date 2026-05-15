"""
MediMind AI — Streamlit frontend.

Multi-page app. Each medical capability has its own page under `pages/`,
and Streamlit auto-detects them. This mirrors the backend's modular
philosophy: adding a page is a self-contained edit.

Run from project root:
    streamlit run frontend/app.py
"""

import os
import sys
from pathlib import Path

# Allow imports like `from frontend.utils import ...` when running streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="MediMind AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Header & disclaimer ----
st.title("🩺 MediMind AI")
st.caption("Modular medical AI assistant — diagnosis, prescription analysis, medical imaging, and more.")

with st.container(border=True):
    st.warning(
        "**Important medical disclaimer**\n\n"
        "MediMind AI is for educational and informational purposes only. "
        "It is **NOT** a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified healthcare "
        "provider with questions about a medical condition. Never disregard "
        "professional medical advice or delay seeking it because of something "
        "you read here.",
        icon="⚠️",
    )

st.markdown("---")

# ---- Backend status ----
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Available capabilities")
    try:
        resp = requests.get(f"{BACKEND}/modules", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        for mod in data.get("modules", []):
            status_emoji = "🟢" if mod.get("status") == "ok" else "🟡"
            st.markdown(
                f"{status_emoji} **{mod['name']}** "
                f"`v{mod['version']}` — status: `{mod.get('status', 'unknown')}`"
            )
    except requests.exceptions.RequestException as e:
        st.error(
            f"Cannot reach backend at `{BACKEND}`. "
            f"Start it with `uvicorn backend.main:app --reload --port 8000`.\n\n"
            f"Details: {e}"
        )

with col2:
    st.subheader("Quick navigation")
    st.page_link("pages/1_Diabetes_Risk.py", label="🩸 Diabetes Risk", icon="➡️")
    st.markdown("_More pages coming as Phase 2/3 ship — see `ROADMAP.md`._")

st.markdown("---")

# ---- About ----
with st.expander("About this project", expanded=False):
    st.markdown(
        """
        **MediMind AI** is a modular, scalable medical AI assistant. Every capability
        (diagnosis, prescription OCR, medical imaging analysis, chat) is a self-contained
        plugin — drop a new folder in `backend/modules/` and it auto-registers.

        **Stack** (all free): FastAPI, scikit-learn, PyTorch, Tesseract, Streamlit,
        Hugging Face Hub, Google Colab.

        See `README.md`, `ARCHITECTURE.md`, and `ROADMAP.md` in the repo for details.
        """
    )
