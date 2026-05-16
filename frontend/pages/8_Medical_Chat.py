"""MediMind RAG chat assistant page."""

import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="MediMind Chat", page_icon="💬", layout="wide")

st.title("💬 MediMind Chat Assistant")
st.caption(
    "General health Q&A grounded in MedlinePlus, WHO, and CDC knowledge sources. "
    "**Not medical advice — always consult a qualified healthcare professional.**"
)

with st.container(border=True):
    st.warning(
        "This assistant provides **general health information only**. "
        "It does NOT give diagnoses, treatment plans, or dosing advice. "
        "In a medical emergency, call emergency services immediately.",
        icon="⚠️",
    )

# ── Status check ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _get_status() -> dict:
    try:
        r = requests.get(f"{BACKEND}/chat_assistant/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

status = _get_status()
kb_ready = status.get("kb_ready", False)
kb_docs = status.get("kb_documents", 0)

if not kb_ready:
    st.error(
        "**Knowledge base is empty — chat assistant not yet ready.**\n\n"
        "To enable it:\n"
        "1. Install Phase 4 dependencies:\n"
        "   ```\n"
        "   pip install sentence-transformers==3.3.1 chromadb==0.5.23 huggingface-hub==0.27.0\n"
        "   ```\n"
        "2. Populate the vector store:\n"
        "   ```\n"
        "   python scripts/ingest_knowledge_base.py\n"
        "   ```\n"
        "3. Set `HUGGINGFACE_TOKEN` in `.env` (free tier) — or run Ollama locally:\n"
        "   ```\n"
        "   ollama pull phi3:mini && ollama serve\n"
        "   ```\n"
        "4. Restart the backend."
    )
    with st.expander("LLM options (both free)"):
        st.markdown(
            "| Option | Setup | Best for |\n"
            "|---|---|---|\n"
            "| HuggingFace Inference API | Set `HUGGINGFACE_TOKEN` in `.env` | Cloud, no GPU needed |\n"
            "| Ollama local | `ollama pull phi3:mini && ollama serve` | Offline / privacy |"
        )
    st.stop()

# ── Sidebar: session info ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Session info")
    st.metric("KB documents", kb_docs)
    llm_opts = status.get("llm_options", [])
    st.caption("LLM options: " + ", ".join(llm_opts) if llm_opts else "Unknown")
    st.markdown("---")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        "**Knowledge sources**\n"
        "- MedlinePlus (NIH, public domain)\n"
        "- WHO fact sheets (open access)\n"
        "- CDC health pages (public domain)"
    )

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a health question…", max_chars=2000)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build history payload (last 10 turns, exclude current message)
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1][-10:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{BACKEND}/chat_assistant/chat",
                    json={"message": user_input, "history": history_payload},
                    timeout=90,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Backend unreachable: {e}")
                st.stop()

        if resp.status_code == 400:
            detail = resp.json().get("detail", "Message flagged.")
            st.error(detail)
            st.session_state.messages.pop()
            st.stop()
        elif resp.status_code != 200:
            st.error(f"Backend error ({resp.status_code}): {resp.text}")
            st.stop()

        data = resp.json()
        answer = data["answer"]
        sources = data.get("sources", [])
        model_used = data.get("model_used", "")
        disclaimer = data.get("disclaimer", "")

        st.markdown(answer)

        if sources:
            with st.expander(f"Sources ({len(sources)} passages retrieved)"):
                for i, src in enumerate(sources, 1):
                    relevance_pct = f"{src['relevance_score']:.0%}"
                    st.markdown(
                        f"**[{i}] {src['title']}** — relevance: `{relevance_pct}`\n\n"
                        f"*{src['source']}*\n\n"
                        f"> {src['excerpt'][:300]}{'…' if len(src['excerpt']) > 300 else ''}"
                    )

        st.caption(f"Model: `{model_used}`")
        st.caption(f"_{disclaimer}_")

    st.session_state.messages.append({"role": "assistant", "content": answer})
