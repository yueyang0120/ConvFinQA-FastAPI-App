"""
Main application module for the Financial QA API.
"""

import os
import sys
from fastapi import FastAPI
from starlette.responses import RedirectResponse

# Import API routers
from src.api.routes import router as finqa_router

# Create FastAPI app
app = FastAPI(
    title="Financial QA API",
    description="API for answering financial questions using an agentic workflow",
    version="1.0.0",
)

# Include routers
app.include_router(finqa_router)

@app.get("/", include_in_schema=False)
def root():
    """Redirect to docs."""
    return RedirectResponse(url="/docs")

@app.get("/health", status_code=200)
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Financial QA API"} 