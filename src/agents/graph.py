"""
LangGraph StateGraph Assembly — src/agents/graph.py

Builds and compiles the multi-agent StateGraph for the GridMind Sentinel system.

Phase 3 graph (5 nodes):
    __start__ → supervisor → fault_analyzer → route_after_analysis →
        LOW:      → report_generator → __end__
        MEDIUM+:  → remediation_agent → guardrail_engine → report_generator → __end__

Nodes:
    - supervisor: Initializes workflow, validates telemetry
    - fault_analyzer: Wavelet analysis + anomaly detection + severity classification
    - remediation_agent: RAG search + action plan generation (MEDIUM+ only)
    - guardrail_engine: Deterministic keyword-based safety validation (MEDIUM+ only)
    - report_generator: Compiles final IncidentReport (always runs)

Connection to system:
    - This is the primary entry point for running fault analysis workflows.
    - Imported by API routes, tests, and the simulation loop.
    - The compiled graph can be invoked with: graph.invoke(initial_state)

Usage:
    from src.agents.graph import build_graph, create_default_graph

    graph = create_default_graph()
    result = graph.invoke({
        "voltage_pu": 0.78,
        "current_pu": 1.45,
        "frequency_hz": 49.5,
        "bus_id": "BUS_001",
        "feeder_id": "F1",
    })
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.fault_analyzer import fault_analyzer_node
from src.agents.guardrails import guardrail_engine_node
from src.agents.remediation import remediation_agent_node
from src.agents.report_generator import report_generator_node
from src.agents.state import GridState
from src.agents.supervisor import route_after_analysis, supervisor_node
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """
    Build the Phase 3 StateGraph (not yet compiled).

    Graph structure:
        __start__ → supervisor → fault_analyzer → route_after_analysis →
            "report_generator"   → report_generator → __end__  (LOW severity)
            "remediation_agent"  → remediation_agent → guardrail_engine
                                 → report_generator → __end__  (MEDIUM+ severity)

    Returns:
        Uncompiled StateGraph instance. Call .compile() before use.
    """
    graph = StateGraph(GridState)

    # ── Add Nodes ──────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("fault_analyzer", fault_analyzer_node)
    graph.add_node("remediation_agent", remediation_agent_node)
    graph.add_node("guardrail_engine", guardrail_engine_node)
    graph.add_node("report_generator", report_generator_node)

    # ── Add Edges ──────────────────────────────────────────────────────────
    # Entry: __start__ → supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor always delegates to fault_analyzer
    graph.add_edge("supervisor", "fault_analyzer")

    # After fault_analyzer: route based on severity
    graph.add_conditional_edges(
        "fault_analyzer",
        route_after_analysis,
        {
            "report_generator": "report_generator",
            "remediation_agent": "remediation_agent",
        },
    )

    # Remediation → Guardrail → Report (always, for MEDIUM+ severity)
    graph.add_edge("remediation_agent", "guardrail_engine")
    graph.add_edge("guardrail_engine", "report_generator")

    # Report generator is always the final node
    graph.add_edge("report_generator", END)

    logger.info(
        "Phase 3 StateGraph built: supervisor → fault_analyzer → "
        "[remediation → guardrail →] report_generator"
    )
    return graph


def create_default_graph():
    """
    Build and compile the default StateGraph for production use.

    Returns:
        Compiled LangGraph that can be invoked with .invoke() or .stream().
    """
    graph = build_graph()
    compiled = graph.compile()
    logger.info("StateGraph compiled successfully")
    return compiled
