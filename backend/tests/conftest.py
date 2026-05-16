"""
Shared pytest fixtures.

Unit tests (schema, service logic) need no fixtures.
Integration tests use `client` which starts the full app with real models.
Mark integration tests with @pytest.mark.integration — they are skipped in
CI environments where models are not present.
"""

import pytest
from fastapi.testclient import TestClient


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires trained models on disk")


@pytest.fixture(scope="session")
def client():
    """Full-app TestClient with lifespan (loads all models once per session)."""
    from backend.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
