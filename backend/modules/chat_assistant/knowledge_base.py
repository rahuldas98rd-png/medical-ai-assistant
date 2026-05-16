"""
RAG knowledge base — ChromaDB vector store + sentence-transformers embeddings.

Knowledge sources (all public domain / free):
  - MedlinePlus health topic summaries (NIH, public domain)
  - WHO fact sheets (open access)
  - CDC health information pages (public domain)

Ingest: python scripts/ingest_knowledge_base.py
"""

from __future__ import annotations

import structlog
from pathlib import Path
from typing import Optional

log = structlog.get_logger()

KB_DIR = Path("data/knowledge_base")
COLLECTION_NAME = "medimind_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 22 MB, runs on CPU, 384-dim embeddings


class KnowledgeBase:
    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._embedder = None
        self._ready = False

    def load(self) -> None:
        """Load ChromaDB collection and embedding model."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            KB_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(KB_DIR / "chroma"))
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._embedder = SentenceTransformer(EMBED_MODEL)
            n_docs = self._collection.count()
            self._ready = n_docs > 0
            log.info(
                "kb.loaded",
                documents=n_docs,
                ready=self._ready,
                hint="Run scripts/ingest_knowledge_base.py to populate." if not self._ready else None,
            )
        except ImportError as e:
            log.warning(
                "kb.import_failed",
                error=str(e),
                hint="Run: pip install sentence-transformers chromadb",
            )
        except Exception as e:
            log.error("kb.load_failed", error=str(e))

    def is_ready(self) -> bool:
        return self._ready

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top_k most relevant passages for the query."""
        if not self.is_ready():
            return []
        query_embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append({
                "excerpt": doc,
                "title": meta.get("title", "Unknown"),
                "source": meta.get("source", ""),
                "relevance_score": float(1.0 - dist),  # cosine distance → similarity
            })
        return docs

    def add_documents(self, documents: list[dict]) -> None:
        """
        Add documents to the collection.
        Each dict must have: id, text, title, source
        """
        if self._collection is None:
            raise RuntimeError("Knowledge base not loaded. Call load() first.")
        embeddings = self._embedder.encode([d["text"] for d in documents]).tolist()
        self._collection.add(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            metadatas=[{"title": d["title"], "source": d["source"]} for d in documents],
            embeddings=embeddings,
        )
        self._ready = self._collection.count() > 0
        log.info("kb.documents_added", count=len(documents), total=self._collection.count())


knowledge_base = KnowledgeBase()
