"""
Custom exceptions for MediMind AI.

Using specific exception types lets the FastAPI exception handler map
them to appropriate HTTP status codes and user-friendly messages,
without leaking internals.
"""


class MediMindError(Exception):
    """Base for all app-specific errors."""

    status_code: int = 500
    user_message: str = "An internal error occurred."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.user_message)
        self.detail = detail or self.user_message


class ModelNotLoadedError(MediMindError):
    """Raised when a module's required model isn't available at inference time."""

    status_code = 503
    user_message = "This diagnostic service is temporarily unavailable. Please try again later."


class InvalidInputError(MediMindError):
    """Raised when input passes Pydantic validation but is semantically invalid."""

    status_code = 422
    user_message = "The provided input is invalid."


class UnsupportedFileTypeError(MediMindError):
    """Raised when a user uploads a file we can't process."""

    status_code = 415
    user_message = "This file type isn't supported."


class ExternalServiceError(MediMindError):
    """Raised when a third-party service (HF API, Ollama) fails."""

    status_code = 502
    user_message = "An external service is currently unreachable."
