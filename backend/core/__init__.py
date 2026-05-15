"""Core infrastructure: module contract, registry, exceptions."""

from backend.core.base_module import BaseModule
from backend.core.registry import registry

__all__ = ["BaseModule", "registry"]
