"""
LangGraph State Schema — src/agents/state.py

Defines the shared TypedDict state that flows through the LangGraph StateGraph.
ALL fields from PRD Section 6 (TelemetryReading, FaultReport, ActionPlan,
IncidentReport) are represented as flat or nested fields in this state dict.

The state is the single source of truth for a fault-handling workflow:
    __start__ → Supervisor → FaultAnalyzer → (Supervisor routing) →
    [RemediationAgent → GuardrailEngine → ReportGenerator] → __end__

Connection to system:
    - Used by graph.py as the StateGraph type parameter.
    - Each agent node reads from and writes to this state.
    - Annotated fields with operator.add enable list accumulation across nodes.

Usage:
    from src.agents.state import GridState
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict


class GridState(TypedDict, total=False):
    """
    LangGraph state for the GridMind Sentinel multi-agent workflow.

    Organized into sections matching PRD Section 6 schemas:
    - Telemetry input fields (TelemetryReading)
    - Fault analysis output fields (FaultReport)
    - Action plan fields (ActionPlan) — populated in Phase 3
    - Incident report fields (IncidentReport) — populated in Phase 3
    - Workflow control fields (routing, iteration, timing)
    """

    # ─── TelemetryReading (PRD Section 6) ─────────────────────────────────
    timestamp: datetime
    bus_id: str                        # e.g., "BUS_001"
    voltage_pu: float                  # per-unit voltage (normal: 0.95–1.05)
    current_pu: float                  # per-unit current (normal: 0.0–1.0)
    frequency_hz: float                # grid frequency (normal: 49.8–50.2)
    active_power_mw: float
    reactive_power_mvar: float
    feeder_id: str

    # ─── FaultReport (PRD Section 6) ──────────────────────────────────────
    fault_id: str                      # UUID
    fault_type: str                    # "voltage_sag" | "overcurrent" |
                                       # "frequency_deviation" | "line_fault" |
                                       # "transformer_overload" | "normal"
    severity: str                      # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    confidence: float                  # 0.0 – 1.0
    affected_components: list[str]
    wavelet_features: dict[str, Any]   # DWT coefficients summary

    # ─── ActionPlan (PRD Section 6) — Phase 3 ────────────────────────────
    plan_id: str                       # UUID
    steps: list[str]                   # Ordered action steps
    requires_switching: bool           # True = needs human approval
    requires_human_approval: bool
    estimated_resolution_time: str
    plan_confidence: float             # Action plan confidence
    references: list[str]             # IEEE/IEC clause references
    guardrail_status: str              # "PASS" | "WARN" | "BLOCK"
    guardrail_reason: str | None

    # ─── IncidentReport (PRD Section 6) — Phase 3 ────────────────────────
    incident_id: str
    agent_trace: Annotated[list[dict[str, Any]], operator.add]  # Accumulates across nodes
    total_latency_ms: int
    llm_tokens_used: int
    resolved: bool
    human_approved: bool | None
    outcome: str | None
    created_at: datetime

    # ─── Workflow Control ─────────────────────────────────────────────────
    next_node: str                     # Routing decision from supervisor
    iteration: int                     # ReAct loop counter (max 3)
    error: str | None                  # Error message if any node fails
