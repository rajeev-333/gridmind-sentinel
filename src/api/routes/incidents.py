"""
Incidents Route — src/api/routes/incidents.py

CRUD operations for incident records.

Endpoints:
    GET  /incidents         — List all incidents (paginated)
    GET  /incidents/{id}    — Get full incident report by ID
    POST /approve/{id}      — Human approval for blocked actions

Connection to system:
    - Reads from IncidentRecord table via SQLAlchemy.
    - Approve endpoint updates requires_human_approval and resolved fields.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.models import IncidentRecord, TelemetryResponse, get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Incidents"])


@router.get("/incidents", response_model=list[TelemetryResponse])
async def list_incidents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    fault_type: Optional[str] = Query(None, description="Filter by fault type"),
):
    """
    List all incidents with optional filtering and pagination.

    Supports filtering by severity (LOW/MEDIUM/HIGH/CRITICAL) and
    fault_type (voltage_sag, overcurrent, etc.).
    """
    try:
        session = get_session()
        query = session.query(IncidentRecord)

        if severity:
            query = query.filter(IncidentRecord.severity == severity.upper())
        if fault_type:
            query = query.filter(IncidentRecord.fault_type == fault_type)

        records = (
            query.order_by(IncidentRecord.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

        results = [record.to_response() for record in records]
        session.close()

        logger.info(f"GET /incidents: returned {len(results)} records")
        return results

    except Exception as e:
        logger.error(f"GET /incidents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/incidents/{incident_id}", response_model=TelemetryResponse)
async def get_incident(incident_id: str):
    """Get a full incident report by incident ID."""
    try:
        session = get_session()
        record = (
            session.query(IncidentRecord)
            .filter(IncidentRecord.incident_id == incident_id)
            .first()
        )
        session.close()

        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Incident {incident_id} not found",
            )

        logger.info(f"GET /incidents/{incident_id}: found")
        return record.to_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /incidents/{incident_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve/{incident_id}")
async def approve_incident(incident_id: str):
    """
    Human approval for a BLOCKED incident.

    Updates the incident record to mark it as human-approved and resolved.
    This endpoint is used when a CRITICAL fault's action plan was blocked
    by the guardrail engine and requires operator authorization.
    """
    try:
        session = get_session()
        record = (
            session.query(IncidentRecord)
            .filter(IncidentRecord.incident_id == incident_id)
            .first()
        )

        if not record:
            session.close()
            raise HTTPException(
                status_code=404,
                detail=f"Incident {incident_id} not found",
            )

        record.human_approved = True
        record.resolved = True
        record.outcome = (
            f"Human-approved: Operator authorized execution of "
            f"blocked remediation plan for {record.fault_type} "
            f"({record.severity})."
        )

        session.commit()
        session.close()

        logger.info(f"POST /approve/{incident_id}: approved")
        return {
            "incident_id": incident_id,
            "status": "approved",
            "message": "Incident approved for execution by human operator",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /approve/{incident_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
