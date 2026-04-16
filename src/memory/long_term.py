"""
Long-Term Incident Memory — src/memory/long_term.py

ChromaDB-based persistent memory for storing and retrieving historical
incident records. Used by the RemediationAgent to find similar past
incidents and by the ReportGenerator to log completed incidents.

Collection: "incident_history"
    - Documents: Incident summary text (fault_type + severity + outcome)
    - Metadata: fault_type, severity, voltage_pu, current_pu, frequency_hz,
                guardrail_status, timestamp
    - IDs: incident_id (UUID)

Connection to system:
    - Called by src/agents/remediation.py to search similar past incidents.
    - Called by src/agents/report_generator.py to store completed incidents.
    - Persisted to data/chroma_db/ via ChromaDB's persistent client.

Usage:
    from src.memory.long_term import IncidentMemory

    memory = IncidentMemory()
    memory.store_incident(incident_id="...", fault_type="line_fault", ...)
    similar = memory.search_similar("line_fault", top_k=3)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ChromaDB storage path
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Singleton client
_client = None
_collection = None


def _get_collection():
    """Get or create the ChromaDB incident_history collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb

        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _collection = _client.get_or_create_collection(
            name="incident_history",
            metadata={"description": "GridMind Sentinel incident history"},
        )
        logger.info(
            f"ChromaDB collection initialized: {_collection.count()} incidents"
        )
        return _collection
    except Exception as e:
        logger.warning(f"ChromaDB initialization failed: {e}")
        return None


class IncidentMemory:
    """
    Long-term memory for incident records using ChromaDB.

    Stores incident summaries as documents with structured metadata.
    Supports similarity search to find past incidents matching a given
    fault type or description.
    """

    def __init__(self):
        """Initialize the incident memory (lazy ChromaDB connection)."""
        self._collection = None

    @property
    def collection(self):
        """Lazy-load the ChromaDB collection."""
        if self._collection is None:
            self._collection = _get_collection()
        return self._collection

    def store_incident(
        self,
        incident_id: str,
        fault_type: str,
        severity: str,
        outcome: str,
        voltage_pu: float = 0.0,
        current_pu: float = 0.0,
        frequency_hz: float = 0.0,
        steps: list[str] | None = None,
        guardrail_status: str = "PASS",
    ) -> None:
        """
        Store a completed incident in long-term memory.

        Args:
            incident_id: Unique incident identifier.
            fault_type: Detected fault type.
            severity: Fault severity level.
            outcome: Resolution outcome description.
            voltage_pu: Voltage measurement at time of fault.
            current_pu: Current measurement at time of fault.
            frequency_hz: Frequency measurement at time of fault.
            steps: Remediation steps taken.
            guardrail_status: Guardrail decision (PASS/WARN/BLOCK).
        """
        collection = self.collection
        if collection is None:
            logger.warning("ChromaDB not available — skipping incident storage")
            return

        # Build document text for embedding-based search
        steps_text = "; ".join(steps) if steps else "No steps"
        document = (
            f"Fault: {fault_type}, Severity: {severity}. "
            f"Telemetry: V={voltage_pu}pu, I={current_pu}pu, f={frequency_hz}Hz. "
            f"Steps: {steps_text}. "
            f"Outcome: {outcome}"
        )

        metadata = {
            "fault_type": fault_type,
            "severity": severity,
            "voltage_pu": voltage_pu,
            "current_pu": current_pu,
            "frequency_hz": frequency_hz,
            "guardrail_status": guardrail_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[incident_id],
            )
            logger.info(
                f"Incident stored in memory: {incident_id[:8]}... "
                f"({fault_type}/{severity})"
            )
        except Exception as e:
            logger.warning(f"Failed to store incident: {e}")

    def search_similar(
        self, fault_type: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Search for similar past incidents by fault type.

        Args:
            fault_type: Fault type to search for.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'document', 'metadata', 'id', and 'distance'.
        """
        collection = self.collection
        if collection is None or collection.count() == 0:
            return []

        try:
            query_text = f"Fault: {fault_type} remediation procedure steps"
            results = collection.query(
                query_texts=[query_text],
                n_results=min(top_k, collection.count()),
            )

            incidents = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    incidents.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i]
                        if results.get("metadatas")
                        else {},
                        "id": results["ids"][0][i]
                        if results.get("ids")
                        else "",
                        "distance": results["distances"][0][i]
                        if results.get("distances")
                        else 0.0,
                    })

            logger.info(
                f"Memory search: {len(incidents)} similar incidents "
                f"found for {fault_type}"
            )
            return incidents

        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    def get_count(self) -> int:
        """Get the number of incidents in memory."""
        collection = self.collection
        return collection.count() if collection else 0
