"""
Base contract for every capability in MediMind AI.

A "module" is a self-contained medical capability: diabetes prediction,
prescription OCR, X-ray classification, etc. By forcing every module to
inherit from BaseModule, we guarantee a uniform interface for:
  - URL routing (each module gets its own prefix)
  - Startup behavior (load models once, not per-request)
  - Health checks (is this module operational?)
  - Discovery (the /modules endpoint lists everything)

To add a new capability, create a class inheriting from BaseModule and
the registry will auto-discover it. See docs/adding_a_module.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fastapi import APIRouter


class BaseModule(ABC):
    """Abstract base for all medical capability modules."""

    # ---- Required class attributes (subclasses MUST override) ----
    name: str = ""                   # URL slug, e.g. "manual_diagnosis"
    version: str = "0.1.0"           # Semver — bump on breaking changes
    description: str = ""            # Human-readable, shown in /modules
    tags: list[str] = []             # OpenAPI grouping, e.g. ["diagnosis"]

    # ---- Optional ----
    requires_models: list[str] = []  # Files in data/models that must exist

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty `name`."
            )
        self._ready: bool = False

    # ---- Required methods ----
    @abstractmethod
    def get_router(self) -> APIRouter:
        """
        Return the FastAPI router for this module.

        The registry mounts it at `{API_PREFIX}/{self.name}`, so a router
        defined with `prefix=""` ends up at e.g. `/api/v1/manual_diagnosis`.
        """
        ...

    # ---- Lifecycle hooks (override as needed) ----
    def on_startup(self) -> None:
        """
        Called once when the FastAPI app starts. Use for expensive setup
        like loading model weights into RAM. Set self._ready = True at
        the end if successful.
        """
        self._ready = True

    def on_shutdown(self) -> None:
        """Called once on app shutdown. Release resources here."""
        pass

    def health_check(self) -> dict[str, Any]:
        """
        Return current health state. The /modules endpoint aggregates
        these. Modules should return enough info to diagnose failures
        (e.g. 'model_file_missing', 'api_unreachable') without leaking
        secrets.
        """
        return {
            "name": self.name,
            "version": self.version,
            "ready": self._ready,
            "status": "ok" if self._ready else "not_ready",
        }

    # ---- Helpers ----
    def __repr__(self) -> str:
        return f"<Module {self.name} v{self.version}>"
