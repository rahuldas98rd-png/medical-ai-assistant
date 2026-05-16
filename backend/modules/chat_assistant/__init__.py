"""
Medical chat assistant module — RAG pipeline with ChromaDB + sentence-transformers.

Phase 4: conversational Q&A grounded in MedlinePlus / WHO / CDC knowledge base.
"""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.chat_assistant.knowledge_base import knowledge_base
from backend.modules.chat_assistant.router import router as chat_router


class ChatAssistantModule(BaseModule):
    name = "chat_assistant"
    version = "0.1.0"
    description = (
        "RAG-powered medical Q&A grounded in trusted public-health knowledge sources "
        "(MedlinePlus, WHO, CDC). Uses ChromaDB vector store + sentence-transformers "
        "embeddings + HuggingFace Inference API / Ollama LLM backend."
    )
    tags = ["chat"]

    def get_router(self) -> APIRouter:
        return chat_router

    def on_startup(self) -> None:
        knowledge_base.load()
        self._ready = knowledge_base.is_ready()

    def health_check(self) -> dict:
        info = super().health_check()
        info["kb_documents"] = (
            knowledge_base._collection.count()
            if knowledge_base._collection else 0
        )
        info["kb_ready"] = knowledge_base.is_ready()
        return info
