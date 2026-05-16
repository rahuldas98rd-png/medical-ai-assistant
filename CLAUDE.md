# Medical AI Assistant — context for Claude Code

## What this project is
Modular medical AI assistant. Plugin-style architecture: every capability
is a folder under `backend/modules/` that inherits from `BaseModule` and
auto-registers at startup. See `ARCHITECTURE.md` and `ROADMAP.md` for the
design rationale and phased plan.

## Current state (May 2026)
- Phase 1: Diabetes risk prediction ✅
- Phase 2: Prescription OCR with fuzzy match + PDF support ✅
- Phase 3: Chest X-ray classifier with Grad-CAM + DICOM + view detection ✅
- Phase 4 (chat assistant) and Phase 5 (orchestrator) are next.

## Hard constraints
- Python 3.12 (NOT 3.13/3.14 — some Phase 3 deps lack wheels)
- All third-party services must be permanently free
- No GPU — training runs on Google Colab, inference is CPU-only
- Windows 11 + PowerShell development environment

## Conventions
- Modules: schemas/ + service.py + router.py + __init__.py with BaseModule subclass
- Pydantic v2 for all schemas
- structlog for logging
- Audit log every prediction to PredictionLog table (hash inputs, never store raw PII)
- Every prediction response includes a `disclaimer` field
- Tests in backend/tests/

Use PowerShell syntax — activate with .\venv\Scripts\Activate.ps1, not source.