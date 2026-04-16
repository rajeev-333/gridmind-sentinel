"""
Telemetry Route — src/api/routes/telemetry.py

Handles telemetry ingestion and triggers the multi-agent workflow.

Endpoints:
    POST /telemetry — Accept a TelemetryReading, run the LangGraph workflow,
                      persist the incident, and return the full result.

Connection to system:
    - Validates input via TelemetryReading Pydantic schema.
    - Invokes the compiled LangGraph (from src/agents/graph.py).
    - Persists the result to SQLite via IncidentRecord.
    - Returns TelemetryResponse with guardrail status.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.models import (
    IncidentRecord,
    TelemetryReading,
    TelemetryResponse,
    get_session,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Telemetry"])

# Lazy-loaded graph to avoid import-time model loading
_compiled_graph = None


def _get_graph():
    """Get or create the compiled LangGraph."""
    global _compiled_graph
    if _compiled_graph is None:
        from src.agents.graph import create_default_graph

        _compiled_graph = create_default_graph()
        logger.info("LangGraph compiled for API use")
    return _compiled_graph


@router.post("/telemetry", response_model=TelemetryResponse)
async def ingest_telemetry(reading: TelemetryReading):
    """
    Ingest a telemetry reading and trigger the multi-agent workflow.

    Runs the full LangGraph pipeline:
        supervisor → fault_analyzer → (routing) →
        [remediation_agent → guardrail_engine →] report_generator

    Returns the complete incident result including fault classification,
    action plan, and guardrail decision.
    """
    start_time = time.time()

    try:
        # Build initial state from telemetry reading
        initial_state = {
            "timestamp": reading.timestamp,
            "bus_id": reading.bus_id,
            "voltage_pu": reading.voltage_pu,
            "current_pu": reading.current_pu,
            "frequency_hz": reading.frequency_hz,
            "active_power_mw": reading.active_power_mw,
            "reactive_power_mvar": reading.reactive_power_mvar,
            "feeder_id": reading.feeder_id,
        }

        logger.info(
            f"POST /telemetry: V={reading.voltage_pu}, "
            f"I={reading.current_pu}, f={reading.frequency_hz}"
        )

        # Run the LangGraph workflow
        graph = _get_graph()
        result = graph.invoke(initial_state)

        # Extract fields from final state
        incident_id = result.get("incident_id", "unknown")
        fault_type = result.get("fault_type", "normal")
        severity = result.get("severity", "LOW")
        confidence = result.get("confidence", 0.0)
        affected = result.get("affected_components", [])
        steps = result.get("steps", [])
        guardrail_status = result.get("guardrail_status", "PASS")
        guardrail_reason = result.get("guardrail_reason")
        requires_human = result.get("requires_human_approval", False)
        est_time = result.get("estimated_resolution_time", "")
        references = result.get("references", [])
        resolved = result.get("resolved", True)
        total_latency = result.get("total_latency_ms", 0)

        # Persist to database
        _persist_incident(
            result=result,
            reading=reading,
            total_latency=total_latency,
        )

        # Build response
        response = TelemetryResponse(
            incident_id=incident_id,
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            affected_components=affected,
            guardrail_status=guardrail_status,
            guardrail_reason=guardrail_reason,
            action_steps=steps,
            requires_human_approval=requires_human,
            estimated_resolution_time=est_time,
            references=references,
            total_latency_ms=total_latency,
            resolved=resolved,
            created_at=result.get("created_at", datetime.now(timezone.utc)),
        )

        logger.info(
            f"POST /telemetry complete: {fault_type}/{severity} "
            f"guardrail={guardrail_status} latency={total_latency}ms"
        )

        return response

    except Exception as e:
        logger.error(f"POST /telemetry failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _persist_incident(
    result: dict,
    reading: TelemetryReading,
    total_latency: int,
) -> None:
    """Persist the incident result to the SQLite database."""
    session = None
    try:
        session = get_session()
        record = IncidentRecord(
            incident_id=result.get("incident_id", "unknown"),
            fault_id=result.get("fault_id", ""),
            fault_type=result.get("fault_type", "normal"),
            severity=result.get("severity", "LOW"),
            confidence=result.get("confidence", 0.0),
            bus_id=reading.bus_id,
            feeder_id=reading.feeder_id,
            voltage_pu=reading.voltage_pu,
            current_pu=reading.current_pu,
            frequency_hz=reading.frequency_hz,
            affected_components=json.dumps(
                result.get("affected_components", [])
            ),
            steps=json.dumps(result.get("steps", [])),
            requires_switching=result.get("requires_switching", False),
            requires_human_approval=result.get(
                "requires_human_approval", False
            ),
            estimated_resolution_time=result.get(
                "estimated_resolution_time", ""
            ),
            plan_confidence=result.get("plan_confidence", 0.0),
            references=json.dumps(result.get("references", [])),
            guardrail_status=result.get("guardrail_status", "PASS"),
            guardrail_reason=result.get("guardrail_reason"),
            total_latency_ms=total_latency,
            resolved=result.get("resolved", True),
            agent_trace=json.dumps(
                result.get("agent_trace", []), default=str
            ),
        )
        session.add(record)
        session.commit()
        logger.debug(f"Incident persisted: {record.incident_id}")
    except Exception as e:
        logger.warning(f"Failed to persist incident: {e}")
        if session:
            session.rollback()
    finally:
        if session:
            session.close()
