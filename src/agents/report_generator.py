"""
Report Generator Agent — src/agents/report_generator.py

LangGraph node that compiles the final IncidentReport from accumulated state.
This is the last node in the workflow before __end__ — it runs for ALL
severity levels (both LOW/normal and MEDIUM+/fault cases).

Responsibilities:
    1. Generate a unique incident_id
    2. Compute total workflow latency
    3. Determine resolution status
    4. Compile all state fields into the IncidentReport structure
    5. Store incident in long-term memory (ChromaDB)

Connection to system:
    - Final node in the LangGraph StateGraph (always runs).
    - Reads ALL accumulated state fields from prior nodes.
    - Writes incident_id, total_latency_ms, resolved, outcome to state.
    - Stores in ChromaDB via src.memory.long_term for future retrieval.

Usage:
    Used as a LangGraph node — not called directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.agents.state import GridState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def report_generator_node(state: GridState) -> dict:
    """
    LangGraph node: Compile the final incident report.

    This node always runs as the last step in the workflow. For LOW severity
    (normal) incidents, it sets default values for fields that were skipped.
    For MEDIUM+ severity incidents, it finalizes the report with all
    accumulated data from remediation and guardrail nodes.

    Args:
        state: Current LangGraph state with all accumulated fields.

    Returns:
        Dict of state updates (IncidentReport fields + agent_trace entry).
    """
    start_time = datetime.now(timezone.utc)

    # Generate incident ID
    incident_id = str(uuid.uuid4())

    # Extract key fields
    fault_type = state.get("fault_type", "normal")
    severity = state.get("severity", "LOW")
    confidence = state.get("confidence", 1.0)
    guardrail_status = state.get("guardrail_status", "PASS")
    steps = state.get("steps", [])
    created_at = state.get("created_at", datetime.now(timezone.utc))

    # Compute total latency from workflow start
    if created_at:
        total_latency_ms = int(
            (datetime.now(timezone.utc) - created_at).total_seconds() * 1000
        )
    else:
        total_latency_ms = 0

    # Determine resolution status
    # Resolved if: normal, or guardrail PASS, or guardrail WARN
    # Not resolved if: guardrail BLOCK (needs human approval)
    resolved = guardrail_status != "BLOCK"

    # Generate outcome description
    outcome = _generate_outcome(fault_type, severity, guardrail_status, resolved)

    # Estimate LLM token usage (template-based approach uses ~0 tokens)
    llm_tokens_used = 0

    # Build trace entry
    elapsed_ms = int(
        (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    )
    trace_entry = {
        "node": "report_generator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "compile_incident_report",
        "inputs": {
            "fault_type": fault_type,
            "severity": severity,
            "guardrail_status": guardrail_status,
            "steps_count": len(steps),
        },
        "outputs": {
            "incident_id": incident_id,
            "resolved": resolved,
            "total_latency_ms": total_latency_ms,
        },
        "elapsed_ms": elapsed_ms,
    }

    # Store in long-term memory
    _store_in_memory(
        incident_id=incident_id,
        fault_type=fault_type,
        severity=severity,
        outcome=outcome,
        state=state,
    )

    logger.info(
        f"ReportGenerator: Incident {incident_id[:8]}... "
        f"({fault_type}/{severity}) resolved={resolved}, "
        f"latency={total_latency_ms}ms"
    )

    # For LOW severity incidents that skipped remediation, set defaults
    result: dict[str, Any] = {
        "incident_id": incident_id,
        "total_latency_ms": total_latency_ms,
        "llm_tokens_used": llm_tokens_used,
        "resolved": resolved,
        "outcome": outcome,
        "agent_trace": [trace_entry],
    }

    # Set defaults for fields that were not populated by skipped nodes
    if severity == "LOW":
        result.update({
            "plan_id": f"noop-{incident_id[:8]}",
            "steps": ["Continue standard monitoring — no action required"],
            "requires_switching": False,
            "requires_human_approval": False,
            "estimated_resolution_time": "N/A",
            "plan_confidence": 1.0,
            "references": [],
            "guardrail_status": "PASS",
            "guardrail_reason": "Normal operation — no guardrail check needed",
        })

    return result


def _generate_outcome(
    fault_type: str,
    severity: str,
    guardrail_status: str,
    resolved: bool,
) -> str:
    """Generate a human-readable outcome description."""
    if fault_type == "normal":
        return "Normal operation — no fault detected, monitoring continues."

    if resolved:
        return (
            f"{fault_type.replace('_', ' ').title()} ({severity}) detected and "
            f"remediation plan approved (guardrail: {guardrail_status}). "
            f"Automated resolution in progress."
        )
    else:
        return (
            f"{fault_type.replace('_', ' ').title()} ({severity}) detected. "
            f"Remediation plan BLOCKED by guardrail — requires human approval "
            f"before switching operations can proceed."
        )


def _store_in_memory(
    incident_id: str,
    fault_type: str,
    severity: str,
    outcome: str,
    state: GridState,
) -> None:
    """Store incident in ChromaDB long-term memory."""
    try:
        from src.memory.long_term import IncidentMemory

        memory = IncidentMemory()
        memory.store_incident(
            incident_id=incident_id,
            fault_type=fault_type,
            severity=severity,
            outcome=outcome,
            voltage_pu=state.get("voltage_pu", 0.0),
            current_pu=state.get("current_pu", 0.0),
            frequency_hz=state.get("frequency_hz", 0.0),
            steps=state.get("steps", []),
            guardrail_status=state.get("guardrail_status", "PASS"),
        )
        logger.debug(f"Incident {incident_id[:8]}... stored in long-term memory")
    except Exception as e:
        logger.warning(f"Failed to store incident in memory: {e}")
