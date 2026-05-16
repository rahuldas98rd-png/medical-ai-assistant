"""
Centralised, tamper-evident prediction logger.

Every call to log_prediction() appends one row to PredictionLog and computes
a chain_hash that incorporates the previous row's hash. Modifying or deleting
any historical row breaks the chain — detectable by GET /api/v1/audit/verify.

Chain formula:
    chain_hash[n] = SHA256(chain_hash[n-1] + "|" + module_name + "|" + input_hash)
    chain_hash[0] = SHA256("genesis|" + module_name + "|" + input_hash)
"""

from __future__ import annotations

import hashlib

import structlog
from sqlalchemy import text

from backend.database import PredictionLog, get_session

log = structlog.get_logger()


def _last_chain_hash(session) -> str:
    row = (
        session.query(PredictionLog.chain_hash)
        .order_by(PredictionLog.id.desc())
        .first()
    )
    if row and row.chain_hash:
        return row.chain_hash
    return "genesis"


def _compute_chain_hash(prev: str, module_name: str, input_hash: str) -> str:
    data = f"{prev}|{module_name}|{input_hash}"
    return hashlib.sha256(data.encode()).hexdigest()


def log_prediction(
    *,
    module_name: str,
    module_version: str,
    input_hash: str,
    prediction: dict,
    confidence: float | None = None,
    latency_ms: int | None = None,
) -> None:
    """Insert one PredictionLog row with a chained integrity hash."""
    try:
        with get_session() as s:
            prev = _last_chain_hash(s)
            chain_hash = _compute_chain_hash(prev, module_name, input_hash)
            s.add(PredictionLog(
                module_name=module_name,
                module_version=module_version,
                input_hash=input_hash,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                chain_hash=chain_hash,
            ))
    except Exception as e:
        log.error("audit_log.write_failed", module=module_name, error=str(e))


def verify_chain() -> dict:
    """
    Read every PredictionLog row in insertion order and recompute the chain.
    Returns {"ok": True, "count": N} or {"ok": False, "first_bad_id": id, "count": N}.
    """
    try:
        with get_session() as s:
            rows = (
                s.query(PredictionLog)
                .order_by(PredictionLog.id.asc())
                .all()
            )
        prev = "genesis"
        for row in rows:
            expected = _compute_chain_hash(prev, row.module_name, row.input_hash)
            if row.chain_hash != expected:
                return {
                    "ok": False,
                    "first_bad_id": row.id,
                    "count": len(rows),
                    "message": (
                        f"Chain broken at log entry id={row.id}. "
                        "This entry or a preceding one may have been tampered with."
                    ),
                }
            prev = row.chain_hash
        return {"ok": True, "count": len(rows), "message": "Audit chain intact."}
    except Exception as e:
        return {"ok": False, "count": 0, "message": f"Verification failed: {e}"}
