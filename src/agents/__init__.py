"""
Multi-Agent Orchestration Package — src/agents/

Implements the LangGraph-based multi-agent system for power grid fault
detection, analysis, and remediation.

Phase 3 Components:
    - state: LangGraph TypedDict state schema (all fields from PRD Section 6)
    - fault_analyzer: FaultAnalyzerAgent with wavelet/anomaly/severity tools
    - supervisor: Supervisor node with severity-based routing logic
    - remediation: RemediationAgent with RAG-based action planning
    - guardrails: Deterministic safety guardrail engine (ZERO LLM calls)
    - report_generator: Incident report compilation + memory storage
    - graph: StateGraph assembly (all 5 nodes with conditional routing)
"""
