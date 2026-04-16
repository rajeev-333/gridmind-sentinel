"""
FastAPI Application — src/api/main.py

Entry point for the GridMind Sentinel REST API. Initializes the FastAPI
application with all route modules, database setup, and middleware.

Endpoints (from PRD Section 5, Feature 7):
    POST /telemetry         — Ingest telemetry, trigger multi-agent workflow
    GET  /incidents         — List all incidents (paginated)
    GET  /incidents/{id}    — Get full incident report
    GET  /metrics           — Return evaluation metrics
    POST /approve/{id}      — Human approval for blocked actions
    GET  /health            — System health check

Connection to system:
    - Registers route blueprints from src/api/routes/
    - Initializes the database on startup via init_db()
    - Serves as the primary interface for external consumers

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models import init_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    # Startup
    logger.info("GridMind Sentinel API starting up...")
    init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("GridMind Sentinel API shutting down...")


app = FastAPI(
    title="GridMind Sentinel API",
    description=(
        "Multi-Agent AI System for Power Grid Monitoring — "
        "Real-time fault detection, RAG-based remediation, "
        "and deterministic safety guardrails."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Route Modules ────────────────────────────────────────────────────
from src.api.routes.telemetry import router as telemetry_router
from src.api.routes.incidents import router as incidents_router
from src.api.routes.metrics import router as metrics_router

app.include_router(telemetry_router)
app.include_router(incidents_router)
app.include_router(metrics_router)


@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "GridMind Sentinel",
        "version": "0.3.0",
    }
