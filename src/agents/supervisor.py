"""
Supervisor Agent — src/agents/supervisor.py

LangGraph node that acts as the orchestrator for the multi-agent workflow.
Receives the initial telemetry event, delegates to FaultAnalyzer, and then
routes based on severity:

Routing logic (from PRD Section 5, Feature 3):
    - LOW severity    → report_generator (monitoring only, no remediation)
    - MEDIUM severity → remediation_agent → guardrail → report_generator
    - HIGH severity   → remediation_agent → guardrail → report_generator
    - CRITICAL        → remediation_agent → guardrail → report_generator
                        (guardrail will BLOCK if switching needed → human approval)

Connection to system:
    - First node after __start__ in the StateGraph.
    - Reads telemetry fields from state, sets up workflow context.
    - After FaultAnalyzer runs, the routing function examines severity
      and sets next_node accordingly.
    - Implements iteration tracking for the ReAct loop (max 3 iterations).

Usage:
    Used as a node in graph.py — not called directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from src.agents.state import GridState
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor Node
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_node(state: GridState) -> dict:
    """
    LangGraph node: Initialize and coordinate the fault analysis workflow.

    Responsibilities:
        1. Validate that telemetry data is present in the state.
        2. Set up workflow metadata (iteration counter, timestamps).
        3. Log the incoming telemetry event.

    This node runs BEFORE the FaultAnalyzer. After FaultAnalyzer completes,
    the `route_after_analysis` function determines the next step.

    Args:
        state: Current LangGraph state with telemetry fields.

    Returns:
        Dict of state updates (workflow control fields + trace entry).
    """
    bus_id = state.get("bus_id", "UNKNOWN")
    feeder_id = state.get("feeder_id", "UNKNOWN")
    voltage_pu = state.get("voltage_pu", 1.0)
    current_pu = state.get("current_pu", 0.5)
    frequency_hz = state.get("frequency_hz", 50.0)
    iteration = state.get("iteration", 0)

    logger.info(
        f"Supervisor: Received telemetry event from {bus_id}/{feeder_id} "
        f"(V={voltage_pu}, I={current_pu}, f={frequency_hz})"
    )

    trace_entry = {
        "node": "supervisor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "initialize_workflow",
        "inputs": {
            "bus_id": bus_id,
            "feeder_id": feeder_id,
            "voltage_pu": voltage_pu,
            "current_pu": current_pu,
            "frequency_hz": frequency_hz,
        },
        "outputs": {
            "iteration": iteration + 1,
            "status": "delegating_to_fault_analyzer",
        },
    }

    return {
        "iteration": iteration + 1,
        "created_at": datetime.now(timezone.utc),
        "agent_trace": [trace_entry],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routing Function (Conditional Edge) — Phase 3
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_analysis(
    state: GridState,
) -> Literal["report_generator", "remediation_agent"]:
    """
    Route the workflow based on fault severity after FaultAnalyzer completes.

    Phase 3 routing rules (PRD Section 5):
        - LOW       → report_generator (monitoring only, skip remediation)
        - MEDIUM    → remediation_agent → guardrail → report_generator
        - HIGH      → remediation_agent → guardrail → report_generator
        - CRITICAL  → remediation_agent → guardrail → report_generator
                      (guardrail may BLOCK, requiring human approval)

    Args:
        state: Current LangGraph state with severity field from FaultAnalyzer.

    Returns:
        Next node name: "report_generator" or "remediation_agent".
    """
    severity = state.get("severity", "LOW")
    fault_type = state.get("fault_type", "normal")
    iteration = state.get("iteration", 0)

    logger.info(
        f"Supervisor routing: {fault_type} ({severity}), iteration={iteration}"
    )

    if severity == "LOW":
        logger.info("Routing: LOW severity → report_generator (monitoring only)")
        return "report_generator"
    elif severity == "MEDIUM":
        logger.info("Routing: MEDIUM severity → remediation_agent")
        return "remediation_agent"
    elif severity == "HIGH":
        logger.info("Routing: HIGH severity → remediation_agent")
        return "remediation_agent"
    elif severity == "CRITICAL":
        logger.info(
            "Routing: CRITICAL severity → remediation_agent "
            "(guardrail will evaluate switching operations)"
        )
        return "remediation_agent"

    # Fallback
    logger.info("Routing: Unknown severity → report_generator (safe default)")
    return "report_generator"
