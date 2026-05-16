"""Schemas for the medical chat assistant."""

from typing import Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=3,
        max_length=2000,
        description="User's medical question.",
        examples=["What are the early signs of type 2 diabetes?"],
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn context (max 10).",
    )


class SourceDocument(BaseModel):
    title: str
    source: str
    excerpt: str = Field(description="Relevant passage from the knowledge base.")
    relevance_score: float = Field(ge=0, le=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = Field(
        description="Knowledge-base passages that grounded this answer."
    )
    model_used: str
    disclaimer: str = (
        "This response is for general educational information only and does NOT "
        "constitute medical advice, diagnosis, or treatment. Always consult a "
        "qualified healthcare professional for personal medical guidance."
    )
