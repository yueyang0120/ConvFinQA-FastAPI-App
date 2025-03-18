"""
Tests for the Financial QA API.
"""

import json
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "Financial QA API"}


def test_root_redirect():
    """Test that the root endpoint redirects to docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# NOTE: Add more detailed API tests here when needed
# They would be integration tests since they would require the LLM to be available 