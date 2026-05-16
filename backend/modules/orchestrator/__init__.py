"""
Orchestrator module — Phase 5.

Routes any combination of symptom text + optional image to the relevant
sub-module services, then synthesizes a unified screening report.
Supported image types: chest X-ray, brain MRI, prescription.
Supported tabular screens: diabetes (needs glucose+bmi+age),
hypertension (needs systolic_bp+age). Heart disease and liver disease
are flagged when keywords match but require the dedicated pages for full
clinical input.
"""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.orchestrator.router import router as orchestrator_router


class OrchestratorModule(BaseModule):
    name = "orchestrator"
    version = "0.1.0"
    description = (
        "Multi-module health screening orchestrator. Accepts free-text symptoms "
        "and an optional medical image, routes to relevant sub-modules, and returns "
        "a unified screening report."
    )
    tags = ["orchestrator"]

    def get_router(self) -> APIRouter:
        return orchestrator_router

    def on_startup(self) -> None:
        self._ready = True

    def health_check(self) -> dict:
        return super().health_check()
