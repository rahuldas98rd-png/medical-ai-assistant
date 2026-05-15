"""
Application configuration.

All settings are loaded from environment variables (or .env file).
Pydantic validates types at startup — bad config fails fast.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "MediMind AI"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:8501,http://localhost:3000"

    # --- Database ---
    database_url: str = "sqlite:///./data/medimind.db"

    # --- Model storage ---
    models_dir: Path = Path("./data/models")

    # --- Third-party (all optional for Phase 1) ---
    huggingface_token: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # --- Frontend ---
    backend_api_url: str = "http://localhost:8000/api/v1"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Use this everywhere — never re-instantiate."""
    return Settings()
