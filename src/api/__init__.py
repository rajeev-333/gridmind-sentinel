"""
API Package — src/api/

FastAPI backend for the GridMind Sentinel system.
Implements all endpoints from PRD Section 5 (Feature 7):

Routes:
    POST /telemetry      — Ingest telemetry, trigger agent workflow
    GET  /incidents       — List all incidents (paginated)
    GET  /incidents/{id}  — Get full incident report
    GET  /metrics         — Return evaluation metrics
    POST /approve/{id}    — Human approval for blocked actions
    GET  /health          — System health check

Components:
    - main: FastAPI application entry point
    - routes/telemetry: Telemetry ingestion and workflow trigger
    - routes/incidents: Incident CRUD operations
    - routes/metrics: System metrics and health checks
"""
