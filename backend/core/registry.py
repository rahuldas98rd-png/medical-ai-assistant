"""
Module discovery and registration.

Walks `backend/modules/`, imports every subpackage, finds every BaseModule
subclass, instantiates it, and exposes the list. The FastAPI app uses this
to mount routers and run lifecycle hooks.

Why auto-discovery? So there's no central import list that goes stale.
You drop a folder under `backend/modules/`, it shows up. That's the
whole extensibility story.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterator

import structlog

from backend.core.base_module import BaseModule

log = structlog.get_logger()


class ModuleRegistry:
    """Singleton-style registry; instantiate once in main.py."""

    def __init__(self) -> None:
        self._modules: dict[str, BaseModule] = {}

    def discover(self, package_name: str = "backend.modules") -> None:
        """
        Import every subpackage under `backend.modules` and register
        any BaseModule subclass found. Idempotent.
        """
        try:
            pkg = importlib.import_module(package_name)
        except ImportError as e:
            log.error("registry.discover.import_failed", package=package_name, error=str(e))
            raise

        for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
            if not ispkg:
                continue  # we only treat subpackages as modules
            full_name = f"{package_name}.{modname}"
            try:
                submod = importlib.import_module(full_name)
            except Exception as e:
                log.warning("registry.discover.skipped", module=full_name, error=str(e))
                continue

            for attr_name in dir(submod):
                attr = getattr(submod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseModule)
                    and attr is not BaseModule
                ):
                    self._register(attr)

    def _register(self, cls: type[BaseModule]) -> None:
        try:
            instance = cls()
        except Exception as e:
            log.error("registry.register.instantiation_failed", cls=cls.__name__, error=str(e))
            return

        if instance.name in self._modules:
            log.warning(
                "registry.register.duplicate",
                name=instance.name,
                existing=self._modules[instance.name].__class__.__name__,
                attempted=cls.__name__,
            )
            return
        self._modules[instance.name] = instance
        log.info("registry.register.ok", module=instance.name, version=instance.version)

    # ---- Iteration & access ----
    def __iter__(self) -> Iterator[BaseModule]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def get(self, name: str) -> BaseModule | None:
        return self._modules.get(name)

    def all(self) -> list[BaseModule]:
        return list(self._modules.values())

    # ---- Lifecycle ----
    def run_startup(self) -> None:
        for m in self._modules.values():
            try:
                m.on_startup()
                log.info("registry.startup.ok", module=m.name)
            except Exception as e:
                log.error("registry.startup.failed", module=m.name, error=str(e))

    def run_shutdown(self) -> None:
        for m in self._modules.values():
            try:
                m.on_shutdown()
            except Exception as e:
                log.error("registry.shutdown.failed", module=m.name, error=str(e))


# Single shared instance, imported by main.py
registry = ModuleRegistry()
