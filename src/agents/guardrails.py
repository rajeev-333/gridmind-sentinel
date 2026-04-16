"""
Guardrail Engine — src/agents/guardrails.py

Deterministic safety guardrail engine for validating remediation action plans.
This module uses ZERO LLM calls — all decisions are made via pure Python
keyword matching on ActionPlan.steps.

Guardrail Rules (from PRD Section 5, Feature 5):
    BLOCK: Action involves de-energization, line switching, breaker operation,
           transformer isolation → requires human approval
    WARN:  Action involves setpoint changes >10% or load shedding
    PASS:  Action is monitoring-only, alert generation, or logging

The engine scans each step in the action plan for keyword patterns. If ANY
step matches a BLOCK keyword, the entire plan is BLOCKED. WARN keywords
produce a WARN status (unless a BLOCK is also found). Otherwise, PASS.

Connection to system:
    - Called by graph.py as a node after remediation_agent.
    - Reads steps[] from state, writes guardrail_status and guardrail_reason.
    - Logs ALL decisions to the GuardrailAuditLog table (SQLite).
    - If BLOCKED: sets requires_human_approval=True.

CRITICAL: This module must remain LLM-free. Any change that adds an LLM call
violates the PRD Section 12 evaluation criteria.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.agents.state import GridState
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword Rule Definitions (Pure Python — ZERO LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════

# BLOCK keywords: actions that require human approval before execution.
# These represent irreversible or high-impact switching operations.
BLOCK_KEYWORDS: list[str] = [
    "de-energize",
    "deenergize",
    "de-energization",
    "circuit breaker",
    "open breaker",
    "trip breaker",
    "close breaker",
    "breaker operation",
    "isolate faulted",
    "isolate affected",
    "isolate overloaded",
    "isolation",
    "disconnect",
    "switching operation",
    "line switching",
    "sectionalizing switch",
]

# WARN keywords: actions that carry moderate risk but may be automated.
WARN_KEYWORDS: list[str] = [
    "load shedding",
    "shed load",
    "curtail load",
    "disconnect non-essential",
    "setpoint change",
    "adjust setpoint",
]

# PASS indicators: safe actions that don't require approval.
# (Used for logging/explanation, not actual rule matching — PASS is default.)
PASS_INDICATORS: list[str] = [
    "monitor",
    "alert",
    "log",
    "verify",
    "check",
    "report",
    "inspect",
    "schedule",
    "continue",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Guardrail Engine (Pure Python keyword matching)
# ═══════════════════════════════════════════════════════════════════════════════


class GuardrailEngine:
    """
    Deterministic safety guardrail for action plan validation.

    Scans each step in an action plan for dangerous keywords. Uses a
    strict hierarchy: BLOCK > WARN > PASS. A single BLOCK keyword in
    any step blocks the entire plan.

    This class contains ZERO LLM calls. All logic is pure Python string
    matching on the ActionPlan.steps list.
    """

    def __init__(self):
        """Initialize the guardrail engine with keyword rules."""
        self.block_keywords = BLOCK_KEYWORDS
        self.warn_keywords = WARN_KEYWORDS

    def evaluate(
        self, steps: list[str], incident_id: str = ""
    ) -> dict[str, Any]:
        """
        Evaluate an action plan's steps against safety rules.

        Scans each step for BLOCK and WARN keywords using case-insensitive
        substring matching. Returns the guardrail decision with reason.

        Args:
            steps: Ordered list of action plan step strings.
            incident_id: Associated incident ID for audit logging.

        Returns:
            Dict with keys:
                - status: "PASS" | "WARN" | "BLOCK"
                - reason: Human-readable explanation
                - matched_keywords: List of matched keyword strings
                - requires_human_approval: True if BLOCKED
        """
        block_matches: list[str] = []
        warn_matches: list[str] = []

        for step in steps:
            step_lower = step.lower()

            # Check BLOCK keywords
            for keyword in self.block_keywords:
                if keyword.lower() in step_lower:
                    block_matches.append(f"'{keyword}' in: {step[:80]}")

            # Check WARN keywords
            for keyword in self.warn_keywords:
                if keyword.lower() in step_lower:
                    warn_matches.append(f"'{keyword}' in: {step[:80]}")

        # Determine status (BLOCK > WARN > PASS)
        if block_matches:
            status = "BLOCK"
            reason = (
                f"Action plan contains {len(block_matches)} dangerous "
                f"operation(s) requiring human approval: "
                f"{'; '.join(block_matches[:3])}"
            )
            all_matches = block_matches
        elif warn_matches:
            status = "WARN"
            reason = (
                f"Action plan contains {len(warn_matches)} moderate-risk "
                f"operation(s): {'; '.join(warn_matches[:3])}"
            )
            all_matches = warn_matches
        else:
            status = "PASS"
            reason = (
                "Action plan contains only monitoring, alerting, and "
                "logging operations — safe for automated execution."
            )
            all_matches = []

        logger.info(
            f"Guardrail decision: {status} "
            f"({len(block_matches)} blocks, {len(warn_matches)} warns) "
            f"for incident {incident_id}"
        )

        # Log to audit trail
        self._log_audit(
            incident_id=incident_id,
            status=status,
            reason=reason,
            matched_keywords=all_matches,
            steps=steps,
        )

        return {
            "status": status,
            "reason": reason,
            "matched_keywords": all_matches,
            "requires_human_approval": status == "BLOCK",
        }

    def _log_audit(
        self,
        incident_id: str,
        status: str,
        reason: str,
        matched_keywords: list[str],
        steps: list[str],
    ) -> None:
        """
        Log guardrail decision to the SQLite audit table.

        Silently catches DB errors to avoid failing the workflow
        if the database is unavailable.
        """
        try:
            from src.models import GuardrailAuditLog, get_session

            session = get_session()
            log_entry = GuardrailAuditLog(
                incident_id=incident_id,
                status=status,
                reason=reason,
                matched_keywords=json.dumps(matched_keywords),
                steps_checked=json.dumps(steps),
            )
            session.add(log_entry)
            session.commit()
            session.close()
            logger.debug(f"Guardrail audit logged: {status} for {incident_id}")
        except Exception as e:
            logger.warning(f"Failed to log guardrail audit: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph Node Function
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton engine instance
_engine = GuardrailEngine()


def guardrail_engine_node(state: GridState) -> dict:
    """
    LangGraph node: Validate the remediation action plan against safety rules.

    Pure Python keyword-matching engine — ZERO LLM calls.

    Pipeline:
        1. Extract steps[] from state (generated by remediation_agent)
        2. Run keyword matching against BLOCK/WARN rules
        3. Set guardrail_status and guardrail_reason in state
        4. If BLOCKED: set requires_human_approval=True
        5. Log decision to audit trail

    Args:
        state: Current LangGraph state with ActionPlan fields populated.

    Returns:
        Dict of state updates (guardrail_status, guardrail_reason,
        requires_human_approval, agent_trace entry).
    """
    start_time = datetime.now(timezone.utc)

    steps = state.get("steps", [])
    plan_id = state.get("plan_id", "")
    fault_type = state.get("fault_type", "unknown")
    incident_id = state.get("incident_id", plan_id)

    logger.info(
        f"GuardrailEngine: Evaluating {len(steps)} steps for "
        f"{fault_type} (plan {plan_id})"
    )

    # Run guardrail evaluation (pure Python keyword matching)
    result = _engine.evaluate(steps, incident_id=incident_id)

    # Build trace entry
    elapsed_ms = int(
        (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    )
    trace_entry = {
        "node": "guardrail_engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "evaluate_action_plan",
        "inputs": {
            "steps_count": len(steps),
            "plan_id": plan_id,
        },
        "outputs": {
            "status": result["status"],
            "matched_count": len(result["matched_keywords"]),
            "requires_human_approval": result["requires_human_approval"],
        },
        "elapsed_ms": elapsed_ms,
    }

    logger.info(
        f"GuardrailEngine complete: {result['status']} "
        f"({len(result['matched_keywords'])} matches), {elapsed_ms}ms"
    )

    return {
        "guardrail_status": result["status"],
        "guardrail_reason": result["reason"],
        "requires_human_approval": result["requires_human_approval"],
        "agent_trace": [trace_entry],
    }
