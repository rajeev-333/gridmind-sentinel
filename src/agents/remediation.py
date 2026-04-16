"""
Remediation Agent — src/agents/remediation.py

LangGraph node that generates remediation action plans for detected faults.
Implements two tool functions as specified in PRD Section 5 (Feature 3):

Tools:
    1. rag_search(query) → list[dict] — Queries the hybrid RAG pipeline for
       relevant IEC/IEEE remediation procedures.
    2. generate_action_plan(state, rag_docs) → dict — Generates an ordered
       action plan based on fault type, severity thresholds, and RAG context.

The action plan generator uses a template-based approach keyed on fault type
and severity. The key design decision is the severity threshold for switching
operations:
    - Moderate faults (V ≥ 0.5 pu, I ≤ 2.0 pu): Conservative steps
      (monitoring, alerts, load adjustment) → guardrail PASS
    - Severe faults (V < 0.5 pu or I > 2.0 pu): Aggressive steps
      (isolation, breaker operation, de-energization) → guardrail BLOCK

This threshold ensures:
    - Scenario A (V=0.78, I=1.4): moderate → PASS
    - Scenario B (V=0.30, I=2.5): severe → BLOCK

Connection to system:
    - Called by graph.py as a node for MEDIUM+ severity faults.
    - Uses src.rag.pipeline.RAGPipeline for standards retrieval.
    - Optionally queries src.memory.long_term for similar past incidents.
    - Output feeds into guardrail_engine for safety validation.

Usage:
    Used as a LangGraph node — not called directly.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from src.agents.state import GridState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialized RAG pipeline (avoids loading models at import time)
_rag_pipeline = None


def _get_rag_pipeline():
    """Get or create the singleton RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from src.rag.pipeline import RAGPipeline

        _rag_pipeline = RAGPipeline()
        _rag_pipeline.initialize()
        logger.info("RAG pipeline initialized for remediation agent")
    return _rag_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: rag_search
# ═══════════════════════════════════════════════════════════════════════════════


def rag_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Query the hybrid RAG pipeline for relevant remediation procedures.

    Searches the IEC/IEEE standards corpus using hybrid retrieval
    (FAISS + BM25 + RRF) with cross-encoder reranking.

    Args:
        query: Natural language query about fault remediation.
        top_k: Number of results to return.

    Returns:
        List of dicts with 'content' and 'metadata' keys.
    """
    try:
        pipeline = _get_rag_pipeline()
        results = pipeline.query(query, top_k=top_k)
        docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
        logger.info(f"RAG search returned {len(docs)} results for: '{query[:60]}...'")
        return docs
    except Exception as e:
        logger.warning(f"RAG search failed: {e}. Using empty context.")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Action Plan Templates (keyed by fault_type × severity level)
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "line_fault": {
        "moderate": [
            "Monitor voltage and current levels on affected bus continuously",
            "Alert grid operations team of detected line fault condition",
            "Verify protective relay settings and coordination on affected feeder",
            "Adjust load distribution on affected feeder (5% rebalancing)",
            "Log fault event and schedule priority inspection of transmission line",
        ],
        "severe": [
            "Immediately isolate faulted line section via protective relay operation",
            "Open circuit breaker on affected feeder to prevent cascading failure",
            "De-energize affected transformer to protect equipment from damage",
            "Dispatch emergency field crew for line inspection and repair",
            "Redirect power flow through alternate transmission path",
        ],
    },
    "voltage_sag": {
        "moderate": [
            "Monitor voltage recovery on affected bus with 1-second sampling",
            "Alert operations center of voltage sag event",
            "Check automatic voltage regulator (AVR) response on affected feeder",
            "Verify tap changer operation on distribution transformer",
            "Log voltage sag duration and depth for power quality records",
        ],
        "severe": [
            "Isolate affected feeder section to prevent equipment damage",
            "Engage backup voltage regulation equipment",
            "De-energize sensitive loads on affected bus",
            "Dispatch field crew to inspect voltage regulation equipment",
            "Restore voltage through alternate supply path",
        ],
    },
    "overcurrent": {
        "moderate": [
            "Monitor current levels with increased sampling frequency",
            "Alert operations team of overcurrent condition",
            "Verify overcurrent relay settings and pickup values",
            "Review demand allocation on overloaded feeder (8% rebalancing)",
            "Log overcurrent event for protection coordination review",
        ],
        "severe": [
            "Trip circuit breaker on overloaded feeder to protect conductors",
            "Isolate faulted section using sectionalizing switches",
            "De-energize affected equipment to prevent thermal damage",
            "Redirect load through alternate feeder paths",
            "Dispatch crew for thermal inspection of conductors and equipment",
        ],
    },
    "frequency_deviation": {
        "moderate": [
            "Monitor frequency trend with 100ms sampling resolution",
            "Alert system operator of frequency deviation event",
            "Verify automatic generation control (AGC) response",
            "Check governor droop settings on connected generators",
            "Log frequency deviation event for grid stability analysis",
        ],
        "severe": [
            "Activate under-frequency load shedding scheme",
            "Disconnect non-essential loads to arrest frequency decline",
            "Engage spinning reserve generators for frequency support",
            "Coordinate with regional grid operator for emergency generation",
            "Monitor frequency recovery and system stability indicators",
        ],
    },
    "transformer_overload": {
        "moderate": [
            "Monitor transformer loading and winding temperature continuously",
            "Alert operations team of transformer overload condition",
            "Rebalance demand on overloaded transformer (5% reallocation)",
            "Check cooling system operation (fans, pumps, radiators)",
            "Log transformer loading for maintenance planning",
        ],
        "severe": [
            "Isolate overloaded transformer to prevent winding damage",
            "Open circuit breaker on high-voltage side of transformer",
            "De-energize transformer to allow cooling before re-energization",
            "Transfer load to backup transformer via switching operation",
            "Dispatch crew for transformer inspection and oil sampling",
        ],
    },
    "normal": {
        "moderate": [
            "Continue standard grid monitoring with normal sampling rates",
            "Log telemetry snapshot for operational records",
            "Verify all protective relay status indicators are normal",
        ],
        "severe": [
            "Continue standard grid monitoring with normal sampling rates",
            "Log telemetry snapshot for operational records",
            "Verify all protective relay status indicators are normal",
        ],
    },
}

# Resolution time estimates
RESOLUTION_TIMES = {
    "line_fault": {"moderate": "15-30 minutes", "severe": "1-4 hours"},
    "voltage_sag": {"moderate": "5-15 minutes", "severe": "30-60 minutes"},
    "overcurrent": {"moderate": "10-20 minutes", "severe": "30-90 minutes"},
    "frequency_deviation": {"moderate": "5-10 minutes", "severe": "15-45 minutes"},
    "transformer_overload": {"moderate": "20-40 minutes", "severe": "2-6 hours"},
    "normal": {"moderate": "N/A", "severe": "N/A"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: generate_action_plan
# ═══════════════════════════════════════════════════════════════════════════════


def _is_severe_fault(
    fault_type: str, voltage_pu: float, current_pu: float
) -> bool:
    """
    Determine if a fault is severe enough to require switching operations.

    This threshold is the key differentiator for guardrail outcomes:
        - Moderate: voltage ≥ 0.5 AND current ≤ 2.0 → conservative steps → PASS
        - Severe:   voltage < 0.5 OR  current > 2.0 → aggressive steps → BLOCK

    Args:
        fault_type: Detected fault type.
        voltage_pu: Per-unit voltage measurement.
        current_pu: Per-unit current measurement.

    Returns:
        True if the fault is severe enough for switching operations.
    """
    if fault_type == "line_fault":
        return voltage_pu < 0.5 or current_pu > 2.0
    if fault_type == "overcurrent":
        return current_pu > 2.0
    if fault_type == "voltage_sag":
        return voltage_pu < 0.5
    if fault_type == "frequency_deviation":
        return False  # Frequency faults handled via generation, not switching
    if fault_type == "transformer_overload":
        return current_pu > 1.8
    return False


def _extract_references(rag_docs: list[dict[str, Any]]) -> list[str]:
    """
    Extract IEC/IEEE clause references from RAG document metadata.

    Parses document source filenames and content to find standard references
    like 'IEC 61850 Clause 7.2' or 'IEEE C37.118 Section 4.1'.

    Args:
        rag_docs: Results from rag_search().

    Returns:
        List of unique reference strings.
    """
    refs = set()
    for doc in rag_docs:
        meta = doc.get("metadata", {})
        source = meta.get("source", "")

        # Extract standard name from filename
        if "iec_61850" in source.lower():
            refs.add("IEC 61850")
        elif "iec_61968" in source.lower():
            refs.add("IEC 61968")
        elif "ieee_c37" in source.lower():
            refs.add("IEEE C37")
        elif "ieee_p2030" in source.lower():
            refs.add("IEEE P2030")

        # Extract clause references from content
        content = doc.get("content", "")
        clause_matches = re.findall(
            r"(?:Clause|Section|Article)\s+[\d]+(?:\.[\d]+)*",
            content,
            re.IGNORECASE,
        )
        refs.update(clause_matches[:3])  # Limit to avoid noise

    return sorted(refs) if refs else ["IEC 61968", "IEEE C37"]


def generate_action_plan(
    state: GridState, rag_docs: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Generate a remediation action plan based on fault analysis and RAG context.

    Selects template steps based on fault type and severity thresholds,
    enriched with references from retrieved IEC/IEEE standards.

    Args:
        state: Current LangGraph state with fault analysis results.
        rag_docs: Retrieved documents from RAG pipeline.

    Returns:
        Dict with ActionPlan fields (plan_id, steps, requires_switching, etc.).
    """
    fault_type = state.get("fault_type", "normal")
    voltage_pu = state.get("voltage_pu", 1.0)
    current_pu = state.get("current_pu", 0.5)
    confidence = state.get("confidence", 0.5)

    # Determine severity level for template selection
    is_severe = _is_severe_fault(fault_type, voltage_pu, current_pu)
    severity_key = "severe" if is_severe else "moderate"

    # Select template
    templates = ACTION_TEMPLATES.get(fault_type, ACTION_TEMPLATES["normal"])
    steps = templates.get(severity_key, templates["moderate"])

    # Extract references from RAG context
    references = _extract_references(rag_docs)

    # Resolution time
    times = RESOLUTION_TIMES.get(fault_type, RESOLUTION_TIMES["normal"])
    est_time = times.get(severity_key, "Unknown")

    # Switching is required for severe faults of certain types
    requires_switching = is_severe and fault_type in (
        "line_fault",
        "overcurrent",
        "transformer_overload",
        "voltage_sag",
    )

    plan = {
        "plan_id": str(uuid.uuid4()),
        "steps": steps,
        "requires_switching": requires_switching,
        "requires_human_approval": requires_switching,
        "estimated_resolution_time": est_time,
        "plan_confidence": round(confidence * 0.9, 4),
        "references": references,
    }

    logger.info(
        f"Action plan generated: {fault_type} ({severity_key}), "
        f"{len(steps)} steps, switching={requires_switching}"
    )

    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph Node Function
# ═══════════════════════════════════════════════════════════════════════════════


def remediation_agent_node(state: GridState) -> dict:
    """
    LangGraph node: Generate a remediation action plan for detected faults.

    Pipeline:
        1. Query RAG pipeline for relevant standards procedures
        2. Optionally check long-term memory for similar past incidents
        3. Generate action plan based on fault type + severity thresholds
        4. Return ActionPlan fields for state update

    Args:
        state: Current LangGraph state with FaultReport fields populated.

    Returns:
        Dict of state updates (ActionPlan fields + agent_trace entry).
    """
    start_time = datetime.now(timezone.utc)

    fault_type = state.get("fault_type", "normal")
    severity = state.get("severity", "LOW")
    bus_id = state.get("bus_id", "UNKNOWN")

    logger.info(
        f"RemediationAgent: Generating plan for {fault_type} ({severity}) "
        f"on {bus_id}"
    )

    # Step 1: RAG search for relevant procedures
    query = _build_rag_query(fault_type, state)
    rag_docs = rag_search(query)

    # Step 2: Check long-term memory for similar incidents (optional)
    past_incidents = _search_past_incidents(fault_type)

    # Step 3: Generate action plan
    plan = generate_action_plan(state, rag_docs)

    # Step 4: Build trace entry
    elapsed_ms = int(
        (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    )
    trace_entry = {
        "node": "remediation_agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "generate_action_plan",
        "inputs": {
            "fault_type": fault_type,
            "severity": severity,
            "rag_docs_count": len(rag_docs),
            "past_incidents_count": len(past_incidents),
        },
        "outputs": {
            "plan_id": plan["plan_id"],
            "steps_count": len(plan["steps"]),
            "requires_switching": plan["requires_switching"],
            "estimated_resolution_time": plan["estimated_resolution_time"],
        },
        "elapsed_ms": elapsed_ms,
    }

    logger.info(
        f"RemediationAgent complete: {len(plan['steps'])} steps, "
        f"switching={plan['requires_switching']}, "
        f"refs={plan['references']}, {elapsed_ms}ms"
    )

    return {
        **plan,
        "agent_trace": [trace_entry],
    }


def _build_rag_query(fault_type: str, state: GridState) -> str:
    """Build a targeted RAG query based on fault type."""
    queries = {
        "line_fault": (
            "line fault remediation procedure isolation switching "
            "protective relay operation transmission line"
        ),
        "voltage_sag": (
            "voltage sag recovery procedure automatic voltage regulator "
            "tap changer compensation"
        ),
        "overcurrent": (
            "overcurrent protection relay coordination "
            "circuit breaker trip settings feeder"
        ),
        "frequency_deviation": (
            "frequency deviation underfrequency protection "
            "automatic generation control governor response"
        ),
        "transformer_overload": (
            "transformer overload protection cooling system "
            "winding temperature load management"
        ),
        "normal": "normal grid operating conditions monitoring procedures",
    }
    return queries.get(fault_type, queries["normal"])


def _search_past_incidents(fault_type: str) -> list[dict]:
    """Search long-term memory for similar past incidents."""
    try:
        from src.memory.long_term import IncidentMemory

        memory = IncidentMemory()
        return memory.search_similar(fault_type, top_k=3)
    except Exception as e:
        logger.debug(f"Past incident search skipped: {e}")
        return []
