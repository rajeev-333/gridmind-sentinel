# GridMind Sentinel — Product Requirements Document (PRD)
**Version:** 1.0  
**Author:** Rajeev Gupta  
**Role:** M.Tech Power Systems Engineering, NIT Warangal  
**Target:** Portfolio project for Agentic AI Engineer II role  
**Date:** April 2026  

---

## 1. Executive Summary

GridMind Sentinel is a production-grade, autonomous multi-agent AI system that monitors
simulated power grid telemetry in real-time, detects faults, retrieves remediation
procedures from IEEE/IEC standards via RAG, and recommends actions — all governed by
enterprise-grade safety guardrails. It demonstrates all 9 key responsibilities of an
Agentic AI Engineer II role.

---

## 2. Problem Statement

Power grid operators must diagnose faults under time pressure using scattered knowledge
from IEC 61968, IEC 61850, and IEEE C37 documents. No open-source portfolio project
currently combines:
- Real-time signal processing with agentic AI
- Domain-specific RAG over power grid standards
- Multi-agent orchestration with safety guardrails
- Built-in evaluation and benchmarking

GridMind Sentinel fills this gap.

---

## 3. System Goals

| Goal | Success Metric |
|------|---------------|
| Detect simulated faults accurately | F1 score ≥ 0.85 on test events |
| Retrieve relevant IEC/IEEE procedures | RAGAS faithfulness ≥ 0.80 |
| Guardrail blocks unsafe actions | 100% of flagged switching ops require human approval |
| Agent task success rate | ≥ 80% autonomous resolution on test scenarios |
| Response latency (P95) | ≤ 8 seconds end-to-end |

---

## 4. User Persona

**Primary User:** Grid Operations Engineer  
- Receives real-time alerts from the system  
- Reviews agent recommendations before approving irreversible actions  
- Monitors system performance via the evaluation dashboard  

**Secondary User:** AI/ML Engineer  
- Reviews agent traces, logs, and evaluation metrics  
- A/B tests agent versions  
- Tunes RAG pipeline and guardrail rules  

---

## 5. Core Features

### Feature 1: Real-Time Grid Telemetry Simulation
- Generate synthetic power grid events: voltage sag, frequency deviation, overcurrent,
  line fault, transformer overload
- Data format: time-series CSV streams (voltage_pu, current_pu, frequency_hz,
  active_power_mw, reactive_power_mvar, timestamp)
- Use PyPSA or manual NumPy simulation (fallback: pre-generated CSV files)
- Events injected at configurable intervals (default: every 10 seconds in demo mode)

### Feature 2: Hybrid RAG Pipeline
- **Document corpus:** IEC 61968 Part 9, IEC 61850, IEEE C37.118, IEEE P2030
  (use publicly available summaries if full docs unavailable)
- **Chunking strategy A:** Recursive character splitter (chunk_size=512, overlap=64)
- **Chunking strategy B:** Semantic sentence splitter (via sentence-transformers)
- **Embedding model:** text-embedding-3-small (OpenAI) or all-MiniLM-L6-v2 (local)
- **Vector store:** FAISS (persistent index saved to disk)
- **Keyword search:** BM25Retriever (rank_bm25 library)
- **Fusion:** Reciprocal Rank Fusion (RRF) combining FAISS + BM25 scores
- **Reranking:** cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)
- **Top-k:** Return top 5 reranked chunks

### Feature 3: Multi-Agent Orchestration (LangGraph)
Build a LangGraph StateGraph with the following nodes:

**Supervisor Agent (Orchestrator)**
- Role: Receives telemetry event → classifies severity (LOW/MEDIUM/HIGH/CRITICAL)
- Routes to: FaultAnalyzerAgent (always) → RemediationAgent (if MEDIUM+) →
  GuardrailEngine (if action proposed) → ReportGeneratorAgent (always)
- Implements ReAct loop: reason → act → observe → repeat (max 3 iterations)
- Uses interrupt_before for CRITICAL events requiring human-in-the-loop

**FaultAnalyzerAgent**
- Tools available:
  - `wavelet_analyze(signal: list[float]) -> dict` — PyWavelets DWT decomposition,
    returns fault_type, confidence, affected_phase
  - `anomaly_score(readings: dict) -> float` — Z-score based anomaly detection
  - `classify_severity(fault_type: str, confidence: float) -> str` — returns
    LOW/MEDIUM/HIGH/CRITICAL
- Output: FaultReport(fault_type, severity, confidence, affected_components, timestamp)

**RemediationAgent**
- Tools available:
  - `rag_search(query: str) -> list[Document]` — queries hybrid RAG pipeline
  - `generate_action_plan(fault_report: FaultReport, docs: list) -> ActionPlan` —
    uses LLM to synthesize recommendations
- Output: ActionPlan(steps: list[str], requires_switching: bool, estimated_time: str,
  references: list[str])

**GuardrailEngine (Node, not LLM-based)**
- Rule-based validation (NOT an LLM call — deterministic logic):
  - BLOCK if: action involves de-energization, line switching, breaker operation,
    transformer isolation
  - WARN if: action involves setpoint changes > 10% or load shedding
  - PASS if: action is monitoring-only, alert generation, logging
- If BLOCKED: set requires_human_approval=True, pause graph (interrupt)
- Log ALL decisions to audit_log table (SQLite)

**ReportGeneratorAgent**
- Generates structured IncidentReport(JSON + Markdown)
- Calls FastAPI endpoint POST /incidents to persist report
- Updates evaluation metrics store

### Feature 4: Memory Architecture
- **Short-term memory:** LangGraph state dict (persists within single incident workflow)
- **Long-term memory:** ChromaDB collection "incident_history"
  - Stores: incident_id, fault_type, action_taken, outcome, timestamp, embeddings
  - Used by RemediationAgent for similar incident retrieval (top-3 past cases)
- **Document memory:** FAISS index (standards corpus, rebuilt on startup)

### Feature 5: Safety Guardrails
- Input guardrail: validate telemetry schema before processing (pydantic models)
- Output guardrail: block/flag dangerous switching recommendations (deterministic rules)
- LLM output guardrail: check for hallucinated IEEE clause numbers (regex + RAG verify)
- Human-in-the-loop: LangGraph interrupt_before node for CRITICAL events
- All guardrail decisions logged to SQLite with reason codes

### Feature 6: Evaluation & Benchmarking Dashboard (Streamlit)
Dashboard tabs:
1. **Live Monitor** — Real-time telemetry feed, active incidents, agent status
2. **Evaluation Metrics** — Task success rate, retrieval faithfulness (RAGAS),
   latency percentiles, guardrail violations, cost per task (token usage)
3. **Agent Traces** — LangSmith-style trace viewer (manual implementation if needed)
4. **A/B Comparison** — Compare agent v1 vs v2 on same test scenario set
5. **Incident History** — Searchable table of past incidents and resolutions

### Feature 7: FastAPI Backend
Endpoints:
- POST /telemetry — ingest new grid reading, trigger agent workflow
- GET /incidents — list all incidents (paginated)
- GET /incidents/{id} — get full incident report
- GET /metrics — return current evaluation metrics
- POST /approve/{incident_id} — human approval for blocked actions
- GET /health — system health check

---

## 6. Data Schemas

### TelemetryReading (Pydantic)
```python
class TelemetryReading(BaseModel):
    timestamp: datetime
    bus_id: str                    # e.g., "BUS_001"
    voltage_pu: float              # per-unit voltage (normal: 0.95-1.05)
    current_pu: float              # per-unit current (normal: 0.0-1.0)
    frequency_hz: float            # grid frequency (normal: 49.8-50.2)
    active_power_mw: float
    reactive_power_mvar: float
    feeder_id: str
```

### FaultReport (Pydantic)
```python
class FaultReport(BaseModel):
    fault_id: str                  # UUID
    timestamp: datetime
    bus_id: str
    fault_type: str                # "voltage_sag" | "overcurrent" | "frequency_deviation" |
                                   # "line_fault" | "transformer_overload" | "normal"
    severity: str                  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    confidence: float              # 0.0 - 1.0
    affected_components: list[str]
    wavelet_features: dict         # DWT coefficients summary
```

### ActionPlan (Pydantic)
```python
class ActionPlan(BaseModel):
    plan_id: str                   # UUID
    fault_id: str
    steps: list[str]               # Ordered action steps
    requires_switching: bool       # True = needs human approval
    requires_human_approval: bool
    estimated_resolution_time: str
    confidence: float
    references: list[str]          # IEEE/IEC clause references
    guardrail_status: str          # "PASS" | "WARN" | "BLOCK"
    guardrail_reason: Optional[str]
```

### IncidentReport (Pydantic)
```python
class IncidentReport(BaseModel):
    incident_id: str
    fault_report: FaultReport
    action_plan: ActionPlan
    agent_trace: list[dict]        # List of agent steps
    total_latency_ms: int
    llm_tokens_used: int
    resolved: bool
    human_approved: Optional[bool]
    outcome: Optional[str]
    created_at: datetime
```

---

## 7. Tech Stack (Exact Versions)

| Component | Library | Version |
|-----------|---------|---------|
| Agent Orchestration | langgraph | 0.2.x |
| LLM Framework | langchain | 0.3.x |
| LLM | openai (gpt-4o-mini) | latest |
| Vector Store | faiss-cpu | 1.8.x |
| Long-term Memory | chromadb | 0.5.x |
| Embeddings | sentence-transformers | 3.x |
| BM25 | rank_bm25 | 0.2.x |
| Reranker | cross-encoder (sentence-transformers) | 3.x |
| RAG Eval | ragas | 0.2.x |
| Signal Processing | PyWavelets | 1.6.x |
| Power Simulation | numpy, scipy | latest |
| Backend API | fastapi + uvicorn | 0.115.x |
| Database | sqlalchemy + sqlite | 2.x |
| Dashboard | streamlit | 1.40.x |
| Data Validation | pydantic | 2.x |
| Visualization | plotly | 5.x |
| Testing | pytest | 8.x |
| Environment | python-dotenv | latest |
| HTTP | httpx | latest |

---

## 8. Project Folder Structure

```
gridmind-sentinel/
├── README.md
├── requirements.txt
├── .env.example
├── pyproject.toml
│
├── data/
│   ├── standards/                 # IEC/IEEE document chunks (text files)
│   ├── simulated/                 # Pre-generated telemetry CSV files
│   └── faiss_index/               # Persisted FAISS index
│
├── src/
│   ├── __init__.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── grid_simulator.py      # Synthetic telemetry generator
│   │   └── fault_injector.py      # Injects fault events into stream
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py     # Load and chunk standards docs
│   │   ├── embeddings.py          # Embedding model wrapper
│   │   ├── vector_store.py        # FAISS + BM25 hybrid retriever
│   │   ├── reranker.py            # Cross-encoder reranker
│   │   └── pipeline.py            # Full RAG pipeline (load → retrieve → rerank)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py               # LangGraph state TypedDict
│   │   ├── supervisor.py          # Supervisor agent node
│   │   ├── fault_analyzer.py      # FaultAnalyzerAgent node + tools
│   │   ├── remediation.py         # RemediationAgent node + tools
│   │   ├── report_generator.py    # ReportGeneratorAgent node
│   │   ├── guardrails.py          # GuardrailEngine (deterministic)
│   │   └── graph.py               # LangGraph StateGraph assembly
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py          # State-based short-term memory
│   │   └── long_term.py           # ChromaDB long-term incident memory
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── telemetry.py
│   │   │   ├── incidents.py
│   │   │   └── metrics.py
│   │   └── models.py              # SQLAlchemy DB models
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Task success, latency, cost, faithfulness
│   │   ├── ragas_eval.py          # RAGAS pipeline evaluation
│   │   └── benchmark.py           # A/B test runner
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Structured logging
│       └── config.py              # Settings from .env
│
├── dashboard/
│   ├── app.py                     # Streamlit main app
│   └── pages/
│       ├── live_monitor.py
│       ├── evaluation.py
│       ├── traces.py
│       └── incidents.py
│
└── tests/
    ├── test_rag.py
    ├── test_agents.py
    ├── test_guardrails.py
    └── test_api.py
```

---

## 9. Build Phases

### Phase 1 (Week 1): RAG Pipeline
1. Create standards corpus (10-20 text files with IEC/IEEE content)
2. Implement document loader with both chunking strategies
3. Build FAISS + BM25 hybrid retriever
4. Add cross-encoder reranker
5. Write test_rag.py — validate faithfulness with 10 sample queries
6. Target: RAGAS faithfulness ≥ 0.75

### Phase 2 (Week 2): Agent Core
1. Define LangGraph state schema (state.py)
2. Implement FaultAnalyzerAgent with wavelet tools
3. Implement Supervisor with ReAct loop
4. Build StateGraph with basic routing
5. Test with 5 simulated fault scenarios
6. Target: Correct fault classification on 4/5 scenarios

### Phase 3 (Week 3): Full System Integration
1. Add RemediationAgent + RAG integration
2. Implement GuardrailEngine (deterministic rules)
3. Add ChromaDB long-term memory
4. Build FastAPI backend (all endpoints)
5. Implement human-in-the-loop interrupt for CRITICAL events
6. Test full pipeline end-to-end

### Phase 4 (Week 4): Evaluation + Polish
1. Build Streamlit dashboard (all 5 tabs)
2. Implement RAGAS evaluation pipeline
3. Write full test suite
4. Add A/B comparison for 2 agent versions
5. Write comprehensive README with architecture diagram
6. Record 3-minute demo video

---

## 10. Demo Scenario Script

For portfolio demonstration, run this scenario:

1. **Normal operation** — 30 seconds of normal telemetry → dashboard shows green
2. **Voltage sag injected** — Agent detects → FaultAnalyzer classifies as MEDIUM →
   RemediationAgent retrieves IEC procedure → GuardrailEngine PASS →
   Action plan generated → Report stored
3. **Line fault injected (CRITICAL)** — GuardrailEngine BLOCKS switching recommendation →
   Human approval prompt appears → Operator approves → Resolution logged
4. **Evaluation tab** — Show live metrics: 2/2 task success (100%), avg latency 4.2s,
   RAGAS faithfulness 0.82, 1 guardrail block (correct)

---

## 11. Out of Scope (For This Version)

- Real SCADA/PMU data integration (use simulation only)
- Multi-user authentication
- Production cloud deployment
- Real-time streaming (use polling at 10s intervals for demo)
- Mobile interface

---

## 12. Evaluation Criteria for the AI Agent Building This

When the AI coding agent builds each module, verify:
- [ ] All Pydantic models match the schemas in Section 6 exactly
- [ ] LangGraph StateGraph has correct node ordering and edge conditions
- [ ] GuardrailEngine uses ZERO LLM calls (must be deterministic rule-based)
- [ ] FAISS index is saved/loaded from disk (not rebuilt each run)
- [ ] All API endpoints return proper HTTP status codes
- [ ] Dashboard connects to FastAPI backend (not hardcoded data)
- [ ] README includes a Mermaid architecture diagram
