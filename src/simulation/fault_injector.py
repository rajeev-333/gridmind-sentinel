"""
Fault Injector — src/simulation/fault_injector.py

Injects controlled fault events into the simulated telemetry stream.
Supports all five fault types from the PRD:
    - voltage_sag: Depresses voltage below 0.90 pu
    - overcurrent: Raises current above 1.2 pu
    - frequency_deviation: Shifts frequency beyond ±0.3 Hz
    - line_fault: Combined voltage collapse + overcurrent
    - transformer_overload: Elevated current + slight voltage depression

Fault configurations are parameterized to support severity gradients:
    - Moderate faults: values near fault thresholds
    - Severe faults: extreme values well beyond thresholds

Connection to system:
    - Used by test scenarios and the API demo endpoint.
    - Takes a normal telemetry reading from GridSimulator and modifies
      it to reflect the specified fault condition.

Usage:
    from src.simulation.fault_injector import FaultInjector

    injector = FaultInjector()
    normal_reading = simulator.generate_reading()
    fault_reading = injector.inject("line_fault", normal_reading, severity="severe")
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Fault Injection Profiles
# ═══════════════════════════════════════════════════════════════════════════════

FAULT_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "voltage_sag": {
        "moderate": {
            "voltage_pu": {"value": 0.82, "noise": 0.02},
        },
        "severe": {
            "voltage_pu": {"value": 0.45, "noise": 0.03},
        },
    },
    "overcurrent": {
        "moderate": {
            "current_pu": {"value": 1.35, "noise": 0.05},
        },
        "severe": {
            "current_pu": {"value": 2.2, "noise": 0.1},
        },
    },
    "frequency_deviation": {
        "moderate": {
            "frequency_hz": {"value": 49.5, "noise": 0.05},
        },
        "severe": {
            "frequency_hz": {"value": 48.8, "noise": 0.1},
        },
    },
    "line_fault": {
        "moderate": {
            "voltage_pu": {"value": 0.78, "noise": 0.02},
            "current_pu": {"value": 1.4, "noise": 0.05},
        },
        "severe": {
            "voltage_pu": {"value": 0.30, "noise": 0.03},
            "current_pu": {"value": 2.5, "noise": 0.1},
        },
    },
    "transformer_overload": {
        "moderate": {
            "current_pu": {"value": 1.25, "noise": 0.03},
            "voltage_pu": {"value": 0.94, "noise": 0.01},
        },
        "severe": {
            "current_pu": {"value": 1.9, "noise": 0.05},
            "voltage_pu": {"value": 0.88, "noise": 0.02},
        },
    },
}


class FaultInjector:
    """
    Controlled fault event injector for simulated telemetry.

    Takes a normal telemetry reading and modifies specific parameters
    to simulate a fault condition. Supports configurable severity levels.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the fault injector.

        Args:
            seed: Random seed for noise generation.
        """
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        fault_type: str,
        reading: dict[str, Any],
        severity: str = "moderate",
    ) -> dict[str, Any]:
        """
        Inject a fault into a telemetry reading.

        Args:
            fault_type: One of "voltage_sag", "overcurrent",
                       "frequency_deviation", "line_fault",
                       "transformer_overload".
            reading: Base telemetry reading dict to modify.
            severity: "moderate" or "severe".

        Returns:
            Modified reading dict with fault values injected.

        Raises:
            ValueError: If fault_type is not recognized.
        """
        if fault_type not in FAULT_PROFILES:
            raise ValueError(
                f"Unknown fault type: {fault_type}. "
                f"Supported: {list(FAULT_PROFILES.keys())}"
            )

        profile = FAULT_PROFILES[fault_type].get(
            severity, FAULT_PROFILES[fault_type]["moderate"]
        )

        fault_reading = copy.deepcopy(reading)

        for param, config in profile.items():
            base_value = config["value"]
            noise = self.rng.normal(0, config.get("noise", 0))
            fault_reading[param] = round(base_value + noise, 4)

        logger.info(
            f"Fault injected: {fault_type} ({severity}) into {reading.get('bus_id', 'UNKNOWN')}"
        )

        return fault_reading

    def inject_custom(
        self,
        reading: dict[str, Any],
        voltage_pu: float | None = None,
        current_pu: float | None = None,
        frequency_hz: float | None = None,
    ) -> dict[str, Any]:
        """
        Inject custom telemetry values (for specific test scenarios).

        Args:
            reading: Base telemetry reading dict.
            voltage_pu: Override voltage value.
            current_pu: Override current value.
            frequency_hz: Override frequency value.

        Returns:
            Modified reading dict.
        """
        fault_reading = copy.deepcopy(reading)

        if voltage_pu is not None:
            fault_reading["voltage_pu"] = voltage_pu
        if current_pu is not None:
            fault_reading["current_pu"] = current_pu
        if frequency_hz is not None:
            fault_reading["frequency_hz"] = frequency_hz

        return fault_reading

    @staticmethod
    def list_fault_types() -> list[str]:
        """Return all supported fault types."""
        return list(FAULT_PROFILES.keys())

    @staticmethod
    def list_severities() -> list[str]:
        """Return all supported severity levels."""
        return ["moderate", "severe"]
