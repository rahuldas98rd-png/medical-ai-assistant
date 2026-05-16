"""Audit module — tamper-evident prediction log verification (Phase 5)."""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.audit.router import router as audit_router


class AuditModule(BaseModule):
    name = "audit"
    version = "0.1.0"
    description = (
        "Tamper-evident audit trail for all model predictions. "
        "Each log entry is chained via SHA-256; verify integrity at GET /audit/verify."
    )
    tags = ["audit"]

    def get_router(self) -> APIRouter:
        return audit_router

    def on_startup(self) -> None:
        self._ready = True
