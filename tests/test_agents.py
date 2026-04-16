"""
Agent Tests — tests/test_agents.py

Phase 2+3 test suite for the multi-agent fault analysis pipeline.

Tests verify:
    1. Voltage sag detection (V=0.82, I=0.6) → voltage_sag, MEDIUM
    2. Overcurrent fault (V=0.95, I=1.5) → overcurrent, HIGH
    3. Line fault (V=0.70, I=1.8) → line_fault, CRITICAL
    4. Frequency deviation (V=1.0, I=0.5, f=48.5) → frequency_deviation
    5. Normal operation (V=1.0, I=0.5, f=50.0) → normal, LOW

Phase 3: Tests also verify remediation, guardrail, and report_generator nodes.

Connection to system:
    - Validates all modules in src/agents/ work correctly together.
    - Run with: pytest tests/test_agents.py -v
"""

from datetime import datetime, timezone

import pytest

from src.agents.fault_analyzer import (
    anomaly_score,
    classify_severity,
    fault_analyzer_node,
    wavelet_analyze,
)
from src.agents.graph import build_graph, create_default_graph
from src.agents.supervisor import route_after_analysis, supervisor_node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compiled_graph():
    """Build and compile the StateGraph once for all tests."""
    return create_default_graph()


# ---------------------------------------------------------------------------
# Test 1: Voltage Sag Scenario
# ---------------------------------------------------------------------------

class TestVoltageSag:
    """Scenario: Voltage drops to 0.82 pu — should detect voltage_sag, MEDIUM."""

    INPUT = {
        "voltage_pu": 0.82,
        "current_pu": 0.6,
        "frequency_hz": 50.0,
        "bus_id": "BUS_002",
        "feeder_id": "F2",
        "timestamp": datetime.now(timezone.utc),
    }

    def test_full_graph_voltage_sag(self, compiled_graph):
        """Full graph should detect voltage sag with MEDIUM severity."""
        result = compiled_graph.invoke(self.INPUT)

        assert result["fault_type"] == "voltage_sag"
        assert result["severity"] in ("MEDIUM", "HIGH")
        assert result["confidence"] > 0.5
        assert "bus" in result["affected_components"] or \
               "voltage_regulator" in result["affected_components"]
        assert result["fault_id"]  # UUID should be set
        # Phase 3: supervisor + fault_analyzer + remediation + guardrail + report
        assert len(result["agent_trace"]) == 5
        # Phase 3: should have remediation + guardrail fields
        assert result.get("guardrail_status") in ("PASS", "WARN", "BLOCK")
        assert result.get("incident_id")  # report_generator sets this

    def test_wavelet_detect_sag_signal(self):
        """Wavelet analysis should detect anomaly in a sagging voltage signal."""
        # Simulate a signal with voltage sag
        signal = [1.0] * 20 + [0.82] * 24 + [1.0] * 20
        result = wavelet_analyze(signal)
        assert result["detail_energy"]  # Should have non-zero detail energy
        assert result["max_detail_energy"] > 0


# ---------------------------------------------------------------------------
# Test 2: Overcurrent Scenario
# ---------------------------------------------------------------------------

class TestOvercurrent:
    """Scenario: Current spikes to 1.5 pu — should detect overcurrent, HIGH."""

    INPUT = {
        "voltage_pu": 0.95,
        "current_pu": 1.5,
        "frequency_hz": 50.0,
        "bus_id": "BUS_003",
        "feeder_id": "F3",
        "timestamp": datetime.now(timezone.utc),
    }

    def test_full_graph_overcurrent(self, compiled_graph):
        """Full graph should detect overcurrent with HIGH severity."""
        result = compiled_graph.invoke(self.INPUT)

        assert result["fault_type"] == "overcurrent"
        assert result["severity"] in ("HIGH", "CRITICAL")
        assert result["confidence"] > 0.6
        assert any(
            c in result["affected_components"]
            for c in ["protective_relay", "feeder"]
        )

    def test_anomaly_score_overcurrent(self):
        """Overcurrent readings should produce high anomaly score."""
        readings = {"voltage_pu": 0.95, "current_pu": 1.5, "frequency_hz": 50.0}
        score = anomaly_score(readings)
        # Current is 1.5, nominal mean is 0.5, std is 0.15
        # Z-score = |1.5 - 0.5| / 0.15 = 6.67
        assert score > 3.0, f"Expected high anomaly score, got {score}"


# ---------------------------------------------------------------------------
# Test 3: Line Fault Scenario (CRITICAL)
# ---------------------------------------------------------------------------

class TestLineFault:
    """Scenario: V=0.70, I=1.8 — simultaneous voltage collapse + overcurrent."""

    INPUT = {
        "voltage_pu": 0.70,
        "current_pu": 1.8,
        "frequency_hz": 49.9,
        "bus_id": "BUS_004",
        "feeder_id": "F4",
        "timestamp": datetime.now(timezone.utc),
    }

    def test_full_graph_line_fault(self, compiled_graph):
        """Full graph should detect line fault with CRITICAL severity."""
        result = compiled_graph.invoke(self.INPUT)

        assert result["fault_type"] == "line_fault"
        assert result["severity"] == "CRITICAL"
        assert result["confidence"] >= 0.85
        assert "transmission_line" in result["affected_components"]

    def test_severity_classification_critical(self):
        """Line fault with high confidence should be CRITICAL."""
        severity = classify_severity("line_fault", 0.92)
        assert severity == "CRITICAL"


# ---------------------------------------------------------------------------
# Test 4: Frequency Deviation Scenario
# ---------------------------------------------------------------------------

class TestFrequencyDeviation:
    """Scenario: f=48.5 Hz — significant under-frequency event."""

    INPUT = {
        "voltage_pu": 1.0,
        "current_pu": 0.5,
        "frequency_hz": 48.5,
        "bus_id": "BUS_005",
        "feeder_id": "F5",
        "timestamp": datetime.now(timezone.utc),
    }

    def test_full_graph_frequency_deviation(self, compiled_graph):
        """Full graph should detect frequency deviation."""
        result = compiled_graph.invoke(self.INPUT)

        assert result["fault_type"] == "frequency_deviation"
        assert result["severity"] in ("MEDIUM", "HIGH", "CRITICAL")
        assert result["confidence"] > 0.5
        assert any(
            c in result["affected_components"]
            for c in ["generator", "load"]
        )

    def test_anomaly_score_frequency(self):
        """Frequency deviation should produce high anomaly score."""
        readings = {"voltage_pu": 1.0, "current_pu": 0.5, "frequency_hz": 48.5}
        score = anomaly_score(readings)
        # |48.5 - 50.0| / 0.05 = 30.0
        assert score > 10.0, f"Expected very high anomaly score, got {score}"


# ---------------------------------------------------------------------------
# Test 5: Normal Operation Scenario
# ---------------------------------------------------------------------------

class TestNormalOperation:
    """Scenario: All readings within normal range — should classify as normal/LOW."""

    INPUT = {
        "voltage_pu": 1.0,
        "current_pu": 0.5,
        "frequency_hz": 50.0,
        "bus_id": "BUS_001",
        "feeder_id": "F1",
        "timestamp": datetime.now(timezone.utc),
    }

    def test_full_graph_normal(self, compiled_graph):
        """Full graph should classify normal readings as normal/LOW."""
        result = compiled_graph.invoke(self.INPUT)

        assert result["fault_type"] == "normal"
        assert result["severity"] == "LOW"
        assert result["confidence"] > 0.5

    def test_anomaly_score_normal(self):
        """Normal readings should produce low anomaly score."""
        readings = {"voltage_pu": 1.0, "current_pu": 0.5, "frequency_hz": 50.0}
        score = anomaly_score(readings)
        assert score < 2.0, f"Expected low anomaly score, got {score}"


# ---------------------------------------------------------------------------
# Test 6: Graph Structure and Routing
# ---------------------------------------------------------------------------

class TestGraphStructure:
    """Tests for graph assembly and routing logic."""

    def test_graph_has_five_nodes(self):
        """Phase 3 graph should have 5 agent nodes."""
        graph = build_graph()
        compiled = graph.compile()
        mermaid = compiled.get_graph().draw_mermaid()
        assert "supervisor" in mermaid
        assert "fault_analyzer" in mermaid
        assert "remediation_agent" in mermaid
        assert "guardrail_engine" in mermaid
        assert "report_generator" in mermaid

    def test_routing_low_severity_to_report(self):
        """LOW severity should route to report_generator (skip remediation)."""
        state = {"severity": "LOW", "fault_type": "normal"}
        result = route_after_analysis(state)  # type: ignore[arg-type]
        assert result == "report_generator"

    def test_routing_medium_severity_to_remediation(self):
        """MEDIUM severity should route to remediation_agent."""
        state = {"severity": "MEDIUM", "fault_type": "voltage_sag"}
        result = route_after_analysis(state)  # type: ignore[arg-type]
        assert result == "remediation_agent"

    def test_routing_critical_severity_to_remediation(self):
        """CRITICAL severity should route to remediation_agent."""
        state = {"severity": "CRITICAL", "fault_type": "line_fault"}
        result = route_after_analysis(state)  # type: ignore[arg-type]
        assert result == "remediation_agent"

    def test_supervisor_increments_iteration(self):
        """Supervisor should increment the iteration counter."""
        state = {"bus_id": "BUS_001", "iteration": 0}
        result = supervisor_node(state)  # type: ignore[arg-type]
        assert result["iteration"] == 1

    def test_agent_trace_accumulates(self, compiled_graph):
        """Agent trace should accumulate entries from all nodes."""
        result = compiled_graph.invoke({
            "voltage_pu": 0.85,
            "current_pu": 0.6,
            "frequency_hz": 50.0,
            "bus_id": "BUS_TEST",
            "feeder_id": "F_TEST",
        })
        trace = result["agent_trace"]
        # Phase 3: MEDIUM severity → 5 nodes (supervisor, fault_analyzer,
        # remediation_agent, guardrail_engine, report_generator)
        assert len(trace) == 5
        assert trace[0]["node"] == "supervisor"
        assert trace[1]["node"] == "fault_analyzer"
        assert trace[2]["node"] == "remediation_agent"
        assert trace[3]["node"] == "guardrail_engine"
        assert trace[4]["node"] == "report_generator"


# ---------------------------------------------------------------------------
# Test 7: Individual Tool Functions
# ---------------------------------------------------------------------------

class TestToolFunctions:
    """Unit tests for the three FaultAnalyzer tools."""

    def test_wavelet_analyze_returns_expected_keys(self):
        """wavelet_analyze should return all expected keys."""
        signal = [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.0] * 4
        result = wavelet_analyze(signal)
        assert "fault_detected" in result
        assert "detail_energy" in result
        assert "max_detail_energy" in result
        assert "approximation_energy" in result
        assert "detail_ratio" in result
        assert "coefficients_summary" in result

    def test_wavelet_analyze_short_signal(self):
        """wavelet_analyze should handle very short signals via padding."""
        result = wavelet_analyze([1.0, 0.5, 1.0])
        assert "detail_energy" in result

    def test_anomaly_score_zero_deviation(self):
        """Perfect nominal readings should give zero anomaly score."""
        readings = {"voltage_pu": 1.0, "current_pu": 0.5, "frequency_hz": 50.0}
        score = anomaly_score(readings)
        assert score == 0.0

    def test_classify_severity_normal(self):
        """Normal fault type should always be LOW."""
        assert classify_severity("normal", 0.95) == "LOW"

    def test_classify_severity_escalation(self):
        """High confidence on HIGH fault type should escalate to CRITICAL."""
        assert classify_severity("overcurrent", 0.92) == "CRITICAL"

    def test_classify_severity_deescalation(self):
        """Low confidence should de-escalate to LOW."""
        assert classify_severity("voltage_sag", 0.30) == "LOW"
