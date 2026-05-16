"""
LLM client — HuggingFace Inference API with Ollama local fallback.

Priority:
  1. HuggingFace Inference API (free tier, cloud) — needs HUGGINGFACE_TOKEN in .env
  2. Ollama local (offline) — needs `ollama pull phi3:mini` and Ollama running

Both paths produce the same interface: generate(prompt) → str
"""

from __future__ import annotations

import structlog

from backend.config import get_settings

log = structlog.get_logger()

HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # free tier on HF Inference API
SAFETY_SUFFIX = (
    "\n\nIMPORTANT: You are a general health information assistant. "
    "Do NOT provide specific dosing, treatment plans, or diagnoses. "
    "Always recommend consulting a qualified healthcare professional."
)
MAX_NEW_TOKENS = 512


def _hf_generate(prompt: str, token: str) -> str:
    """Call HuggingFace Inference API."""
    import requests as req
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.3,
            "do_sample": True,
            "return_full_text": False,
        },
    }
    resp = req.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()
    raise ValueError(f"Unexpected HF API response: {data}")


def _ollama_generate(prompt: str, base_url: str, model: str) -> str:
    """Call local Ollama API."""
    import requests as req
    resp = req.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate(prompt: str) -> tuple[str, str]:
    """
    Generate a response. Returns (answer_text, model_name_used).
    Tries HF first, falls back to Ollama, raises if both unavailable.
    """
    settings = get_settings()
    safe_prompt = prompt + SAFETY_SUFFIX

    if settings.huggingface_token:
        try:
            answer = _hf_generate(safe_prompt, settings.huggingface_token)
            log.info("llm.generated", backend="huggingface", model=HF_MODEL)
            return answer, f"HF/{HF_MODEL}"
        except Exception as e:
            log.warning("llm.hf_failed", error=str(e), fallback="ollama")

    # Ollama fallback
    try:
        answer = _ollama_generate(safe_prompt, settings.ollama_base_url, settings.ollama_model)
        log.info("llm.generated", backend="ollama", model=settings.ollama_model)
        return answer, f"Ollama/{settings.ollama_model}"
    except Exception as e:
        log.error("llm.ollama_failed", error=str(e))
        raise RuntimeError(
            "No LLM backend available. "
            "Either set HUGGINGFACE_TOKEN in .env or start Ollama locally "
            "(ollama pull phi3:mini && ollama serve)."
        ) from e


def build_rag_prompt(question: str, context_docs: list[dict], history: list[dict]) -> str:
    """Assemble the RAG prompt from retrieved passages + conversation history."""
    ctx_block = "\n\n".join(
        f"[{i+1}] {d['title']} ({d['source']}):\n{d['excerpt']}"
        for i, d in enumerate(context_docs)
    ) or "No specific knowledge-base passages found for this query."

    history_block = ""
    for turn in history[-4:]:   # keep last 4 turns to stay within context
        role = "User" if turn["role"] == "user" else "Assistant"
        history_block += f"{role}: {turn['content']}\n"

    return (
        "You are MediMind, a medical information assistant. "
        "Answer the user's question using ONLY the provided context passages. "
        "If the context doesn't cover the question, say so clearly. "
        "Never give specific treatment advice or dosing.\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"CONVERSATION:\n{history_block}"
        f"User: {question}\n"
        "Assistant:"
    )
