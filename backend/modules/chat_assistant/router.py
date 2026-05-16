"""Routes for the medical chat assistant module."""

from fastapi import APIRouter, HTTPException

from backend.modules.chat_assistant.knowledge_base import knowledge_base
from backend.modules.chat_assistant.llm_client import build_rag_prompt, generate
from backend.modules.chat_assistant.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SourceDocument,
)

router = APIRouter()

SAFETY_BLOCKLIST = [
    "suicide", "self-harm", "overdose how to", "kill myself",
]


def _is_safe(text: str) -> bool:
    lower = text.lower()
    return not any(kw in lower for kw in SAFETY_BLOCKLIST)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Medical Q&A grounded in trusted health knowledge base",
)
async def chat(req: ChatRequest) -> ChatResponse:
    if not _is_safe(req.message):
        raise HTTPException(
            status_code=400,
            detail=(
                "This message has been flagged. If you are in crisis, please contact "
                "a mental health professional or emergency services immediately."
            ),
        )

    if len(req.history) > 10:
        req.history = req.history[-10:]

    # RAG retrieval
    context_docs = knowledge_base.retrieve(req.message, top_k=3)

    # Build prompt and generate
    prompt = build_rag_prompt(
        req.message,
        context_docs,
        [m.model_dump() for m in req.history],
    )
    try:
        answer, model_used = generate(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sources = [SourceDocument(**doc) for doc in context_docs]

    return ChatResponse(answer=answer, sources=sources, model_used=model_used)


@router.get("/status", summary="Chat assistant readiness status")
async def status() -> dict:
    return {
        "kb_ready": knowledge_base.is_ready(),
        "kb_documents": (
            knowledge_base._collection.count()
            if knowledge_base._collection else 0
        ),
        "embed_model": "all-MiniLM-L6-v2",
        "llm_options": ["HuggingFace Inference API", "Ollama local"],
        "setup_steps": [
            "1. pip install sentence-transformers chromadb (already in requirements.txt)",
            "2. python scripts/ingest_knowledge_base.py  (populates ChromaDB)",
            "3. Set HUGGINGFACE_TOKEN in .env  OR  run: ollama pull phi3:mini && ollama serve",
        ],
    }
