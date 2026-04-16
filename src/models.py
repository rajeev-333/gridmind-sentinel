"""
Data Models — src/models.py

Pydantic schemas for API request/response validation and SQLAlchemy ORM models
for database persistence (SQLite).

Pydantic schemas implement the exact data contracts from PRD Section 6:
    - TelemetryReading: Input validation for telemetry data
    - FaultReportSchema: Fault analysis output
    - ActionPlanSchema: Remediation plan output
    - IncidentReportSchema: Complete incident record
    - TelemetryResponse: API response for POST /telemetry

SQLAlchemy models for persistent storage:
    - IncidentRecord: Full incident lifecycle record
    - GuardrailAuditLog: Deterministic guardrail decision audit trail

Connection to system:
    - Pydantic schemas used by src/api/routes/ for request/response validation.
    - SQLAlchemy models used by report_generator and API routes for persistence.
    - Database initialized via init_db() on application startup.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config import PROJECT_ROOT, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas (API Request / Response)
# ═══════════════════════════════════════════════════════════════════════════════


class TelemetryReading(BaseModel):
    """Input schema for grid telemetry data (PRD Section 6)."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    bus_id: str = "BUS_001"
    voltage_pu: float = Field(..., description="Per-unit voltage (normal: 0.95–1.05)")
    current_pu: float = Field(..., description="Per-unit current (normal: 0.0–1.0)")
    frequency_hz: float = Field(..., description="Grid frequency Hz (normal: 49.8–50.2)")
    active_power_mw: float = 100.0
    reactive_power_mvar: float = 30.0
    feeder_id: str = "F1"


class TelemetryResponse(BaseModel):
    """Response schema for POST /telemetry — full incident summary."""

    incident_id: str
    fault_type: str
    severity: str
    confidence: float
    affected_components: list[str] = []
    guardrail_status: str
    guardrail_reason: Optional[str] = None
    action_steps: list[str] = []
    requires_human_approval: bool = False
    estimated_resolution_time: str = ""
    references: list[str] = []
    total_latency_ms: int = 0
    resolved: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IncidentListItem(BaseModel):
    """Compact schema for GET /incidents list."""

    incident_id: str
    fault_type: str
    severity: str
    guardrail_status: str
    resolved: bool
    created_at: datetime


class MetricsResponse(BaseModel):
    """Response schema for GET /metrics."""

    total_incidents: int = 0
    avg_latency_ms: float = 0.0
    guardrail_pass_count: int = 0
    guardrail_warn_count: int = 0
    guardrail_block_count: int = 0
    fault_type_distribution: dict[str, int] = {}
    severity_distribution: dict[str, int] = {}
    resolution_rate: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SQLAlchemy ORM Models
# ═══════════════════════════════════════════════════════════════════════════════


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


class IncidentRecord(Base):
    """
    Persistent incident record — stores the full lifecycle of a grid fault.

    Created by the API layer after the LangGraph workflow completes.
    Tracks fault analysis, remediation plan, guardrail decision, and resolution.
    """

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String, unique=True, index=True, nullable=False)
    fault_id = Column(String)
    fault_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    confidence = Column(Float)
    bus_id = Column(String)
    feeder_id = Column(String)
    voltage_pu = Column(Float)
    current_pu = Column(Float)
    frequency_hz = Column(Float)
    affected_components = Column(Text, default="[]")  # JSON string
    steps = Column(Text, default="[]")  # JSON string
    requires_switching = Column(Boolean, default=False)
    requires_human_approval = Column(Boolean, default=False)
    estimated_resolution_time = Column(String, default="")
    plan_confidence = Column(Float, default=0.0)
    references = Column(Text, default="[]")  # JSON string
    guardrail_status = Column(String, default="PASS")
    guardrail_reason = Column(Text, nullable=True)
    total_latency_ms = Column(Integer, default=0)
    resolved = Column(Boolean, default=False)
    human_approved = Column(Boolean, nullable=True)
    outcome = Column(Text, nullable=True)
    agent_trace = Column(Text, default="[]")  # JSON string
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    def to_response(self) -> TelemetryResponse:
        """Convert DB record to API response schema."""
        return TelemetryResponse(
            incident_id=self.incident_id,
            fault_type=self.fault_type,
            severity=self.severity,
            confidence=self.confidence or 0.0,
            affected_components=json.loads(self.affected_components or "[]"),
            guardrail_status=self.guardrail_status or "PASS",
            guardrail_reason=self.guardrail_reason,
            action_steps=json.loads(self.steps or "[]"),
            requires_human_approval=self.requires_human_approval or False,
            estimated_resolution_time=self.estimated_resolution_time or "",
            references=json.loads(self.references or "[]"),
            total_latency_ms=self.total_latency_ms or 0,
            resolved=self.resolved or False,
            created_at=self.created_at or datetime.now(timezone.utc),
        )


class GuardrailAuditLog(Base):
    """
    Audit trail for guardrail decisions — logs ALL decisions (PASS/WARN/BLOCK).

    Required by PRD Section 5 Feature 5: All guardrail decisions logged to
    SQLite with reason codes. This table provides a complete audit trail
    for compliance and debugging.
    """

    __tablename__ = "guardrail_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String, index=True, nullable=False)
    status = Column(String, nullable=False)  # PASS | WARN | BLOCK
    reason = Column(Text, nullable=True)
    matched_keywords = Column(Text, default="[]")  # JSON string
    steps_checked = Column(Text, default="[]")  # JSON string
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Database Engine & Session Management
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None
_SessionLocal = None


def get_engine():
    """
    Get or create the SQLAlchemy engine (singleton).

    Handles the sqlite:///data/gridmind.db → absolute path conversion
    so the DB file is always created relative to the project root.
    """
    global _engine
    if _engine is not None:
        return _engine

    db_url = settings.DATABASE_URL

    # Convert relative sqlite path to absolute
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        rel_path = db_url.replace("sqlite:///", "")
        abs_path = PROJECT_ROOT / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{abs_path}"

    _engine = create_engine(db_url, echo=False)
    logger.info(f"Database engine created: {db_url}")
    return _engine


def get_session() -> Session:
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db() -> None:
    """Create all database tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables initialized")
