"""
Database engine, session factory, and table definitions.

SQLite for development (zero setup). Switch to Postgres/Supabase by
changing DATABASE_URL in .env — the rest of the code doesn't care.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
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
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db() -> None:
    """Create all tables. Called on app startup."""
    # Ensure parent directory exists for SQLite
    if settings.database_url.startswith("sqlite"):
        from pathlib import Path
        db_path = settings.database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)


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
