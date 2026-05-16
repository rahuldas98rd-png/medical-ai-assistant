"""
MediMind AI — FastAPI entry point.

Boots the app, configures logging, discovers modules under `backend/modules/`,
and mounts each one's router. Adding a new capability means creating a folder
under `backend/modules/` — nothing in this file changes.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.config import get_settings
from backend.core.auth import ApiKeyMiddleware
from backend.core.exceptions import MediMindError
from backend.core.logging_setup import configure_logging
from backend.core.rate_limiter import limiter
from backend.core.registry import registry
from backend.database import init_db

settings = get_settings()
configure_logging()
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown handlers."""
    log.info("app.starting", env=settings.app_env, debug=settings.debug)
    init_db()
    log.info("db.initialized", url=settings.database_url.split("@")[-1])  # don't log creds

    # Discover and start modules
    registry.discover()
    registry.run_startup()
    log.info("modules.loaded", count=len(registry), names=[m.name for m in registry])

    # Mount each module's router under the API prefix
    for module in registry:
        app.include_router(
            module.get_router(),
            prefix=f"{settings.api_prefix}/{module.name}",
            tags=module.tags or [module.name],
        )
        log.info("module.mounted", name=module.name, prefix=f"{settings.api_prefix}/{module.name}")

    yield

    log.info("app.stopping")
    registry.run_shutdown()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Modular medical AI assistant. **For educational/informational use only — "
        "this is NOT a substitute for professional medical advice, diagnosis, or treatment.**"
    ),
    version="0.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Rate limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middleware (order matters: outermost runs first on request) ---
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global exception handler ---
@app.exception_handler(MediMindError)
async def medimind_error_handler(request: Request, exc: MediMindError) -> JSONResponse:
    log.warning("error.handled", type=type(exc).__name__, detail=exc.detail, path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "message": exc.user_message, "detail": exc.detail},
    )


# --- Top-level routes ---
@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "modules_endpoint": f"{settings.api_prefix}/modules",
        "disclaimer": (
            "This service is for educational purposes only. Always consult a "
            "qualified healthcare professional for medical advice."
        ),
    }


@app.get(f"{settings.api_prefix}/modules", tags=["meta"])
async def list_modules() -> dict:
    """List every registered module and its health status."""
    return {
        "count": len(registry),
        "modules": [m.health_check() for m in registry],
    }


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
