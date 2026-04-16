"""
Grid Simulator — src/simulation/grid_simulator.py

Generates synthetic power grid telemetry data for testing and demonstration.
Produces realistic time-series readings that follow nominal operating conditions
with configurable noise and drift patterns.

Normal operating ranges (from PRD Section 6):
    - voltage_pu:    0.95 – 1.05 (mean: 1.0, std: 0.02)
    - current_pu:    0.0  – 1.0  (mean: 0.5, std: 0.15)
    - frequency_hz:  49.8 – 50.2 (mean: 50.0, std: 0.05)
    - active_power_mw:    50 – 200 MW
    - reactive_power_mvar: 10 – 60 MVAR

Connection to system:
    - Used by src/simulation/fault_injector.py for base telemetry generation.
    - Used by API demo routes to generate streaming telemetry data.
    - Can export to CSV files in data/simulated/ for offline testing.

Usage:
    from src.simulation.grid_simulator import GridSimulator

    sim = GridSimulator(num_buses=5)
    reading = sim.generate_reading(bus_id="BUS_001")
    batch = sim.generate_batch(num_readings=100)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GridSimulator:
    """
    Synthetic power grid telemetry generator.

    Produces realistic telemetry readings with configurable noise,
    drift, and bus/feeder topology. Each bus has slightly different
    nominal values to simulate a realistic multi-bus grid.
    """

    # Nominal operating parameters
    NOMINAL = {
        "voltage_pu": {"mean": 1.0, "std": 0.015, "min": 0.95, "max": 1.05},
        "current_pu": {"mean": 0.5, "std": 0.10, "min": 0.1, "max": 0.9},
        "frequency_hz": {"mean": 50.0, "std": 0.03, "min": 49.85, "max": 50.15},
        "active_power_mw": {"mean": 120.0, "std": 15.0, "min": 50.0, "max": 200.0},
        "reactive_power_mvar": {"mean": 35.0, "std": 8.0, "min": 10.0, "max": 60.0},
    }

    def __init__(
        self,
        num_buses: int = 5,
        feeders_per_bus: int = 2,
        seed: int = 42,
    ):
        """
        Initialize the grid simulator.

        Args:
            num_buses: Number of simulated buses.
            feeders_per_bus: Number of feeders per bus.
            seed: Random seed for reproducibility.
        """
        self.num_buses = num_buses
        self.feeders_per_bus = feeders_per_bus
        self.rng = np.random.default_rng(seed)

        # Generate bus and feeder IDs
        self.buses = [f"BUS_{i+1:03d}" for i in range(num_buses)]
        self.feeders = {
            bus: [f"F{bus[-3:]}_{j+1}" for j in range(feeders_per_bus)]
            for bus in self.buses
        }

        # Per-bus nominal offsets (slight variations to simulate topology)
        self.bus_offsets = {
            bus: {
                "voltage_pu": self.rng.normal(0, 0.005),
                "current_pu": self.rng.normal(0, 0.03),
                "frequency_hz": 0.0,  # Frequency is system-wide
                "active_power_mw": self.rng.normal(0, 10),
                "reactive_power_mvar": self.rng.normal(0, 3),
            }
            for bus in self.buses
        }

        logger.info(
            f"GridSimulator initialized: {num_buses} buses, "
            f"{feeders_per_bus} feeders each"
        )

    def generate_reading(
        self,
        bus_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Generate a single telemetry reading for a bus.

        Args:
            bus_id: Target bus ID. Random if not specified.
            timestamp: Reading timestamp. Now if not specified.

        Returns:
            Dict matching TelemetryReading schema.
        """
        if bus_id is None:
            bus_id = self.rng.choice(self.buses)

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        offsets = self.bus_offsets.get(bus_id, {k: 0 for k in self.NOMINAL})
        feeder_id = self.rng.choice(
            self.feeders.get(bus_id, ["F1"])
        )

        reading = {
            "timestamp": timestamp,
            "bus_id": bus_id,
            "feeder_id": feeder_id,
        }

        for param, nom in self.NOMINAL.items():
            value = self.rng.normal(
                nom["mean"] + offsets.get(param, 0),
                nom["std"],
            )
            # Clip to nominal range
            value = float(np.clip(value, nom["min"], nom["max"]))
            reading[param] = round(value, 4)

        return reading

    def generate_batch(
        self,
        num_readings: int = 100,
        interval_seconds: float = 10.0,
        bus_id: str | None = None,
        start_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a batch of time-series telemetry readings.

        Args:
            num_readings: Number of readings to generate.
            interval_seconds: Time between consecutive readings.
            bus_id: Target bus (random rotation if None).
            start_time: Start timestamp for the batch.

        Returns:
            List of telemetry reading dicts.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        readings = []
        for i in range(num_readings):
            ts = start_time + timedelta(seconds=i * interval_seconds)
            reading = self.generate_reading(bus_id=bus_id, timestamp=ts)
            readings.append(reading)

        logger.info(f"Generated batch: {num_readings} readings")
        return readings

    def export_csv(
        self,
        readings: list[dict[str, Any]],
        filename: str = "telemetry.csv",
    ) -> Path:
        """
        Export readings to a CSV file in data/simulated/.

        Args:
            readings: List of telemetry reading dicts.
            filename: Output filename.

        Returns:
            Path to the created CSV file.
        """
        output_dir = settings.simulated_data_path
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        if not readings:
            logger.warning("No readings to export")
            return filepath

        fieldnames = list(readings[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for reading in readings:
                row = {
                    k: v.isoformat() if isinstance(v, datetime) else v
                    for k, v in reading.items()
                }
                writer.writerow(row)

        logger.info(f"Exported {len(readings)} readings to {filepath}")
        return filepath
