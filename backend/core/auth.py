"""
API key authentication middleware.

Dev mode (API_KEY unset): all requests pass through.
Prod mode (API_KEY set in .env): every POST/PUT/PATCH requires X-API-Key header.
GET endpoints (status, health, docs) are always open.

Usage in main.py:
    app.add_middleware(ApiKeyMiddleware)
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config import get_settings

# Paths that are always open regardless of auth setting
_OPEN_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json", "/")
_OPEN_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        # Auth disabled — dev mode
        if not settings.api_key:
            return await call_next(request)

        # Always open: safe HTTP methods and specific paths
        if request.method in _OPEN_METHODS:
            return await call_next(request)

        if any(request.url.path == p or request.url.path.startswith(p + "/")
               for p in _OPEN_PREFIXES if p != "/"):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid or missing API key. Add X-API-Key header.",
                },
            )

        return await call_next(request)
