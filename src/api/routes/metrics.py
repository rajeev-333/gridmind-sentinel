"""
Metrics Route — src/api/routes/metrics.py

System evaluation metrics and health monitoring endpoints.

Endpoints:
    GET /metrics — Return aggregate evaluation metrics computed from
                   incident history in the database.

Connection to system:
    - Queries IncidentRecord table for aggregate statistics.
    - Returns MetricsResponse schema.
"""

from __future__ import annotations

from sqlalchemy import func

from fastapi import APIRouter, HTTPException

from src.models import IncidentRecord, MetricsResponse, get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Return aggregate evaluation metrics for the system.

    Computes statistics from the incident history:
        - Total incidents processed
        - Average latency (ms)
        - Guardrail decision distribution (PASS/WARN/BLOCK)
        - Fault type distribution
        - Severity distribution
        - Resolution rate
    """
    try:
        session = get_session()

        # Total incidents
        total = session.query(func.count(IncidentRecord.id)).scalar() or 0

        if total == 0:
            session.close()
            return MetricsResponse()

        # Average latency
        avg_latency = (
            session.query(func.avg(IncidentRecord.total_latency_ms)).scalar()
            or 0.0
        )

        # Guardrail distribution
        guardrail_pass = (
            session.query(func.count(IncidentRecord.id))
            .filter(IncidentRecord.guardrail_status == "PASS")
            .scalar()
            or 0
        )
        guardrail_warn = (
            session.query(func.count(IncidentRecord.id))
            .filter(IncidentRecord.guardrail_status == "WARN")
            .scalar()
            or 0
        )
        guardrail_block = (
            session.query(func.count(IncidentRecord.id))
            .filter(IncidentRecord.guardrail_status == "BLOCK")
            .scalar()
            or 0
        )

        # Fault type distribution
        fault_types = (
            session.query(
                IncidentRecord.fault_type,
                func.count(IncidentRecord.id),
            )
            .group_by(IncidentRecord.fault_type)
            .all()
        )
        fault_dist = {ft: count for ft, count in fault_types}

        # Severity distribution
        severities = (
            session.query(
                IncidentRecord.severity,
                func.count(IncidentRecord.id),
            )
            .group_by(IncidentRecord.severity)
            .all()
        )
        severity_dist = {sev: count for sev, count in severities}

        # Resolution rate
        resolved_count = (
            session.query(func.count(IncidentRecord.id))
            .filter(IncidentRecord.resolved == True)
            .scalar()
            or 0
        )
        resolution_rate = (resolved_count / total * 100) if total > 0 else 0.0

        session.close()

        metrics = MetricsResponse(
            total_incidents=total,
            avg_latency_ms=round(avg_latency, 1),
            guardrail_pass_count=guardrail_pass,
            guardrail_warn_count=guardrail_warn,
            guardrail_block_count=guardrail_block,
            fault_type_distribution=fault_dist,
            severity_distribution=severity_dist,
            resolution_rate=round(resolution_rate, 1),
        )

        logger.info(f"GET /metrics: {total} incidents, {resolution_rate:.0f}% resolved")
        return metrics

    except Exception as e:
        logger.error(f"GET /metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
