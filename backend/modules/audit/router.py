"""Audit routes — tamper-evident log verification."""

from fastapi import APIRouter

from backend.core.audit_log import verify_chain
from backend.database import PredictionLog, get_session

router = APIRouter()


@router.get(
    "/verify",
    summary="Verify the prediction audit log chain integrity",
)
async def verify() -> dict:
    """
    Recomputes the SHA-256 chain across every PredictionLog row.
    Returns ok=True if the chain is intact, or identifies the first
    tampered/deleted entry by id.
    """
    return verify_chain()


@router.get(
    "/stats",
    summary="Prediction log statistics",
)
async def stats() -> dict:
    """Count total predictions per module."""
    try:
        with get_session() as s:
            rows = s.query(
                PredictionLog.module_name,
                PredictionLog.prediction,
            ).all()

        counts: dict[str, dict] = {}
        for module_name, prediction in rows:
            task = prediction.get("task", "unknown") if isinstance(prediction, dict) else "unknown"
            key = f"{module_name}/{task}"
            counts[key] = counts.get(key, 0) + 1

        return {
            "total_predictions": len(rows),
            "by_task": counts,
        }
    except Exception as e:
        return {"error": str(e)}
