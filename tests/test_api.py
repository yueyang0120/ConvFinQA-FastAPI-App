"""
Tests for the Financial QA API.
"""

import json
from fastapi.testclient import TestClient

from agentic_financial_qa.main import app

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


def test_openapi_schema_includes_financial_qa_route():
    """Test that the public API contract exposes the financial QA endpoint."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/financial-qa/questions" in response.json()["paths"]
