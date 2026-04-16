"""
Fault Analyzer Agent — src/agents/fault_analyzer.py

LangGraph node that analyzes power grid telemetry to detect and classify faults.
Implements three tool functions as specified in PRD Section 5 (Feature 3):

Tools:
    1. wavelet_analyze(signal) — PyWavelets DWT decomposition → fault_type,
       confidence, affected_phase
    2. anomaly_score(readings) — Z-score based anomaly detection
    3. classify_severity(fault_type, confidence) — Maps fault + confidence
       to LOW/MEDIUM/HIGH/CRITICAL

The node function `fault_analyzer_node` orchestrates all three tools:
    1. Synthesizes a signal from telemetry readings
    2. Runs wavelet analysis to identify fault type
    3. Computes anomaly score for additional confidence signal
    4. Classifies severity based on fault type and composite confidence
    5. Writes FaultReport fields into the LangGraph state

Connection to system:
    - Called by graph.py as a node in the StateGraph.
    - Reads telemetry fields (voltage_pu, current_pu, frequency_hz) from state.
    - Writes fault_type, severity, confidence, wavelet_features, etc. to state.
    - Supervisor routes based on the severity output from this node.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pywt

from src.agents.state import GridState
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Nominal operating ranges (from PRD Section 6 — TelemetryReading)
# ═══════════════════════════════════════════════════════════════════════════════

NOMINAL = {
    "voltage_pu": {"min": 0.95, "max": 1.05, "mean": 1.0, "std": 0.02},
    "current_pu": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.15},
    "frequency_hz": {"min": 49.8, "max": 50.2, "mean": 50.0, "std": 0.05},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: wavelet_analyze
# ═══════════════════════════════════════════════════════════════════════════════

def wavelet_analyze(signal: list[float]) -> dict[str, Any]:
    """
    Perform Discrete Wavelet Transform (DWT) decomposition on a signal.

    Uses the Daubechies-4 (db4) wavelet at level 3 to extract detail
    coefficients. High energy in detail coefficients indicates transient
    disturbances (faults).

    Args:
        signal: Time-series signal values (e.g., voltage samples).

    Returns:
        Dict with keys:
            - fault_detected (bool): Whether the signal shows fault characteristics.
            - detail_energy (list[float]): Energy in each detail coefficient level.
            - max_detail_energy (float): Maximum energy across detail levels.
            - approximation_energy (float): Energy in the approximation coefficients.
            - coefficients_summary (dict): Statistics of DWT coefficients.
    """
    sig = np.array(signal, dtype=np.float64)

    # Pad signal to minimum length for DWT if needed (2^level = 8 minimum)
    min_len = 16
    if len(sig) < min_len:
        # Repeat signal to reach minimum length
        repeats = (min_len // len(sig)) + 1
        sig = np.tile(sig, repeats)[:min_len]

    # DWT decomposition: db4 wavelet, level 3
    wavelet = "db4"
    level = min(3, pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(sig, wavelet, level=level)

    # coeffs[0] = approximation, coeffs[1:] = detail levels (coarsest to finest)
    approx_energy = float(np.sum(coeffs[0] ** 2))
    detail_energies = [float(np.sum(c ** 2)) for c in coeffs[1:]]
    max_detail = max(detail_energies) if detail_energies else 0.0

    # Fault detection: high detail energy relative to approximation
    total_energy = approx_energy + sum(detail_energies)
    detail_ratio = sum(detail_energies) / total_energy if total_energy > 0 else 0.0
    fault_detected = detail_ratio > 0.15  # >15% energy in details = anomalous

    return {
        "fault_detected": fault_detected,
        "detail_energy": detail_energies,
        "max_detail_energy": max_detail,
        "approximation_energy": approx_energy,
        "detail_ratio": float(detail_ratio),
        "coefficients_summary": {
            "wavelet": wavelet,
            "level": level,
            "num_coefficients": sum(len(c) for c in coeffs),
            "signal_length": len(sig),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: anomaly_score
# ═══════════════════════════════════════════════════════════════════════════════

def anomaly_score(readings: dict[str, float]) -> float:
    """
    Compute a composite anomaly score using Z-score based detection.

    Calculates how far each reading deviates from nominal operating conditions
    (in standard deviations), then returns the maximum Z-score across all
    monitored parameters.

    Args:
        readings: Dict with keys "voltage_pu", "current_pu", "frequency_hz".
                  Each value is the current measurement.

    Returns:
        Composite anomaly score (float). Higher = more anomalous.
        Typical thresholds: <2.0 normal, 2.0-3.0 warning, >3.0 fault.
    """
    z_scores = {}

    for param, value in readings.items():
        if param in NOMINAL:
            nom = NOMINAL[param]
            z = abs(value - nom["mean"]) / nom["std"] if nom["std"] > 0 else 0.0
            z_scores[param] = float(z)

    # Composite: maximum z-score across parameters
    # This catches the single worst deviation
    composite = max(z_scores.values()) if z_scores else 0.0

    logger.info(
        f"Anomaly scores: {', '.join(f'{k}={v:.2f}' for k, v in z_scores.items())} "
        f"-> composite={composite:.2f}"
    )

    return composite


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: classify_severity
# ═══════════════════════════════════════════════════════════════════════════════

def classify_severity(fault_type: str, confidence: float) -> str:
    """
    Classify fault severity based on fault type and detection confidence.

    Severity levels (from PRD):
        - LOW: Minor deviation, monitoring-only response
        - MEDIUM: Moderate fault, automated remediation possible
        - HIGH: Significant fault, requires coordinated response
        - CRITICAL: Severe fault, requires human-in-the-loop approval

    Args:
        fault_type: Detected fault type string.
        confidence: Detection confidence (0.0 to 1.0).

    Returns:
        Severity string: "LOW", "MEDIUM", "HIGH", or "CRITICAL".
    """
    # Base severity by fault type (some faults are inherently more severe)
    base_severity = {
        "normal": "LOW",
        "voltage_sag": "MEDIUM",
        "frequency_deviation": "MEDIUM",
        "overcurrent": "HIGH",
        "transformer_overload": "HIGH",
        "line_fault": "CRITICAL",
    }

    severity = base_severity.get(fault_type, "MEDIUM")

    # Escalate based on confidence
    if confidence >= 0.9 and severity == "HIGH":
        severity = "CRITICAL"
    elif confidence >= 0.85 and severity == "MEDIUM":
        severity = "HIGH"
    elif confidence < 0.4 and severity in ("MEDIUM", "HIGH"):
        severity = "LOW"

    logger.info(
        f"Severity classification: {fault_type} (conf={confidence:.2f}) "
        f"-> {severity}"
    )

    return severity


# ═══════════════════════════════════════════════════════════════════════════════
# Fault type identification from telemetry
# ═══════════════════════════════════════════════════════════════════════════════

def _identify_fault_type(
    voltage_pu: float,
    current_pu: float,
    frequency_hz: float,
    wavelet_result: dict[str, Any],
) -> tuple[str, float, list[str]]:
    """
    Identify fault type from telemetry values and wavelet analysis.

    Uses a rule-based decision tree informed by power systems domain knowledge:
    - Voltage sag: voltage < 0.90 pu
    - Overcurrent: current > 1.2 pu
    - Frequency deviation: |frequency - 50.0| > 0.3 Hz
    - Line fault: voltage < 0.85 AND current > 1.3 (simultaneous)
    - Transformer overload: current > 1.1 AND voltage slightly depressed

    Args:
        voltage_pu: Per-unit voltage measurement.
        current_pu: Per-unit current measurement.
        frequency_hz: Frequency measurement in Hz.
        wavelet_result: Output from wavelet_analyze().

    Returns:
        Tuple of (fault_type, confidence, affected_components).
    """
    affected = []
    fault_scores: dict[str, float] = {}

    # Line fault: simultaneous voltage collapse + overcurrent (most severe)
    if voltage_pu < 0.85 and current_pu > 1.3:
        fault_scores["line_fault"] = 0.90 + min(0.10, (1.3 - voltage_pu) * 0.2)
        affected.extend(["transmission_line", "bus", "protective_relay"])

    # Voltage sag
    if voltage_pu < 0.90:
        # Deeper sag = higher confidence
        sag_depth = (0.90 - voltage_pu) / 0.90
        fault_scores["voltage_sag"] = min(0.95, 0.60 + sag_depth * 2.0)
        if "bus" not in affected:
            affected.append("bus")
        affected.append("voltage_regulator")

    # Overcurrent
    if current_pu > 1.2:
        excess = (current_pu - 1.0) / 1.0
        fault_scores["overcurrent"] = min(0.95, 0.55 + excess * 1.5)
        if "protective_relay" not in affected:
            affected.append("protective_relay")
        affected.append("feeder")

    # Frequency deviation
    freq_dev = abs(frequency_hz - 50.0)
    if freq_dev > 0.3:
        fault_scores["frequency_deviation"] = min(0.95, 0.50 + freq_dev * 1.0)
        affected.append("generator")
        affected.append("load")

    # Transformer overload: moderate overcurrent + slight voltage depression
    if current_pu > 1.1 and voltage_pu < 0.97 and "line_fault" not in fault_scores:
        fault_scores["transformer_overload"] = min(
            0.90, 0.55 + (current_pu - 1.0) * 1.0
        )
        affected.append("transformer")

    # Boost confidence if wavelet confirms fault
    if wavelet_result.get("fault_detected", False):
        for ft in fault_scores:
            fault_scores[ft] = min(1.0, fault_scores[ft] + 0.05)

    # Select highest-confidence fault type
    if fault_scores:
        fault_type = max(fault_scores, key=fault_scores.get)  # type: ignore[arg-type]
        confidence = fault_scores[fault_type]
    else:
        fault_type = "normal"
        confidence = 1.0 - wavelet_result.get("detail_ratio", 0.0)
        confidence = max(0.5, confidence)

    return fault_type, round(confidence, 4), affected


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph Node Function
# ═══════════════════════════════════════════════════════════════════════════════

def fault_analyzer_node(state: GridState) -> dict:
    """
    LangGraph node: Analyze telemetry and produce a FaultReport.

    Pipeline:
        1. Synthesize signal from telemetry readings
        2. Run wavelet_analyze() for transient detection
        3. Compute anomaly_score() for z-score based detection
        4. Identify fault type using rule-based classifier
        5. Classify severity using classify_severity()
        6. Return FaultReport fields for state update

    Args:
        state: Current LangGraph state with telemetry fields populated.

    Returns:
        Dict of state updates (FaultReport fields + agent_trace entry).
    """
    start_time = datetime.now(timezone.utc)

    voltage_pu = state.get("voltage_pu", 1.0)
    current_pu = state.get("current_pu", 0.5)
    frequency_hz = state.get("frequency_hz", 50.0)
    bus_id = state.get("bus_id", "UNKNOWN")

    logger.info(
        f"FaultAnalyzer: Analyzing telemetry for {bus_id} "
        f"(V={voltage_pu}, I={current_pu}, f={frequency_hz})"
    )

    # Step 1: Synthesize signal for wavelet analysis
    # In production, this would be a time-series buffer. Here we create a
    # representative signal centered around the measurement with injected noise.
    rng = np.random.default_rng(seed=42)
    base_signal = np.full(64, voltage_pu)
    noise = rng.normal(0, 0.01, 64)
    # Add transient if voltage is abnormal
    if voltage_pu < 0.90 or voltage_pu > 1.10:
        transient = np.zeros(64)
        transient[20:30] = (1.0 - voltage_pu) * 0.5  # Inject transient dip/swell
        base_signal = base_signal + transient
    signal = (base_signal + noise).tolist()

    # Step 2: Wavelet analysis
    wavelet_result = wavelet_analyze(signal)

    # Step 3: Anomaly score
    readings = {
        "voltage_pu": voltage_pu,
        "current_pu": current_pu,
        "frequency_hz": frequency_hz,
    }
    a_score = anomaly_score(readings)

    # Step 4: Identify fault type
    fault_type, confidence, affected = _identify_fault_type(
        voltage_pu, current_pu, frequency_hz, wavelet_result
    )

    # Step 5: Classify severity
    severity = classify_severity(fault_type, confidence)

    # Step 6: Build trace entry
    elapsed_ms = int(
        (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    )
    trace_entry = {
        "node": "fault_analyzer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "analyze_telemetry",
        "inputs": {
            "voltage_pu": voltage_pu,
            "current_pu": current_pu,
            "frequency_hz": frequency_hz,
            "bus_id": bus_id,
        },
        "outputs": {
            "fault_type": fault_type,
            "severity": severity,
            "confidence": confidence,
            "anomaly_score": a_score,
            "wavelet_fault_detected": wavelet_result["fault_detected"],
        },
        "elapsed_ms": elapsed_ms,
    }

    logger.info(
        f"FaultAnalyzer result: {fault_type} ({severity}) "
        f"conf={confidence:.2f}, anomaly={a_score:.2f}"
    )

    return {
        "fault_id": str(uuid.uuid4()),
        "fault_type": fault_type,
        "severity": severity,
        "confidence": confidence,
        "affected_components": affected,
        "wavelet_features": {
            "detail_energy": wavelet_result["detail_energy"],
            "max_detail_energy": wavelet_result["max_detail_energy"],
            "detail_ratio": wavelet_result["detail_ratio"],
            "fault_detected": wavelet_result["fault_detected"],
            "anomaly_score": a_score,
        },
        "agent_trace": [trace_entry],
    }
