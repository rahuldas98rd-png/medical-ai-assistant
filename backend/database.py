"""
Database engine, session factory, and table definitions.

SQLite for development (zero setup). Switch to Postgres/Supabase by
changing DATABASE_URL in .env — the rest of the code doesn't care.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings

settings = get_settings()

# `check_same_thread=False` is required for SQLite + FastAPI (multiple threads).
# It's a no-op for Postgres.
_engine_kwargs = (
    {"connect_args": {"check_same_thread": False}}
    if settings.database_url.startswith("sqlite")
    else {}
)
engine = create_engine(settings.database_url, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class for ORM models."""


class PredictionLog(Base):
    """
    Audit trail of every prediction made by any module.

    We store input_hash (NOT raw input) for privacy — a hash lets us
    detect duplicate predictions and debug aggregate behavior without
    keeping patient data around.
    """

    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_name = Column(String(64), index=True, nullable=False)
    module_version = Column(String(16), nullable=False)
    input_hash = Column(String(64), index=True, nullable=False)  # SHA-256 hex
    prediction = Column(JSON, nullable=False)                    # arbitrary module output
    confidence = Column(Float, nullable=True)                    # optional, [0,1]
    latency_ms = Column(Integer, nullable=True)
    # Tamper-evident chain: SHA256(prev_chain_hash | module_name | input_hash)
    chain_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ConsultationHistory(Base):
    """
    One row per orchestrator /analyze call — stores the full report for
    per-user history. user_key is a short hash of the API key or client IP;
    it never stores the raw value.
    """

    __tablename__ = "consultation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_key = Column(String(16), index=True, nullable=False)   # first 16 hex chars of SHA-256
    symptoms_preview = Column(String(200), nullable=True)       # first 200 chars of symptoms
    overall_summary = Column(Text, nullable=True)
    report_json = Column(JSON, nullable=False)                   # full OrchestratorReport dict
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db() -> None:
    """Create all tables. Called on app startup."""
    # Ensure parent directory exists for SQLite
    if settings.database_url.startswith("sqlite"):
        from pathlib import Path
        db_path = settings.database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)
    _migrate()


def _migrate() -> None:
    """Apply additive schema changes to existing tables (SQLite-safe)."""
    with engine.connect() as conn:
        # Add chain_hash to prediction_log if it doesn't exist yet
        cols = [r[1] for r in conn.execute(sa_text("PRAGMA table_info(prediction_log)"))]
        if "chain_hash" not in cols:
            conn.execute(sa_text(
                "ALTER TABLE prediction_log ADD COLUMN chain_hash VARCHAR(64)"
            ))
            conn.commit()


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager for ad-hoc DB work (not via FastAPI dependency)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
